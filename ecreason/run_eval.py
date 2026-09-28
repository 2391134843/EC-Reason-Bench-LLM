"""Cascade evaluator: walk the EC tree level-by-level with a model, then score.

The QUERY is always the enzyme sequence (shown unless --no-seq) plus its intrinsic
sequence-derived features. Retrieval EVIDENCE channels are toggled via --channels,
which is how we run ABLATIONS:

  closed-book (sequence + features only):   --channels none
  + ESM-kNN:                                --channels esm_knn
  + DIAMOND BLAST homology:                 --channels blast
  + structure (Foldseek):                   --channels structure
  + active-site (Folddisco):                --channels active_site
  + Pfam domains:                           --channels pfam
  full open-book (everything available):    --channels all     (or --open-book)
  blind (no sequence, no evidence):         --channels none --no-seq

Modes:
  cascade (default) - candidates at level L = children of the model's chosen node at L-1
                      (errors compound; realistic). static - teacher-forced gold parent.

Backends: 'mock' (no API/GPU), 'openai:<model>' [API], 'hf:<model_id>' [GPU].
"""
from __future__ import annotations

import argparse
import json
import random
import time

from .build_items import SPLITS, _letters
from .ec_tree import ECTree, UNSURE, build_candidates, node_at_level, split_ec
from .models import build_backend
from .paths import EVIDENCE_DIR, ITEMS_DIR, RESULTS_DIR, ensure_out_dirs
from .prompts import build_step_prompt
from .scoring import evaluate

ALL_CHANNELS = ["esm_knn", "blast", "hmmer", "structure", "active_site", "pfam", "seq_homology"]
# evidence file suffix per channel ('' = the main {split}.jsonl which holds 3 channels)
_CHANNEL_FILE = {
    "esm_knn": "", "structure": "", "active_site": "",
    "blast": ".blast", "hmmer": ".hmmer", "pfam": ".pfam", "seq_homology": ".kmer",
}


def load_items(split: str) -> list[dict]:
    with open(ITEMS_DIR / f"{split}.jsonl") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def subsample(items: list[dict], limit: int, split: str) -> list[dict]:
    """Representative subset of size `limit`, seeded by split name so EVERY method/condition
    evaluates the SAME items (keeps conditions comparable).

    Plain ``items[:limit]`` is biased: the CARE test CSVs are EC-sorted, so the first rows are
    all the same EC main class (e.g. EC 1). A seeded random sample restores class diversity.
    """
    if not limit or limit >= len(items):
        return items
    rng = random.Random(f"ecreason-sample:{split}")
    idx = sorted(rng.sample(range(len(items)), limit))
    return [items[i] for i in idx]


def load_evidence(split: str) -> dict:
    """Merge every available evidence channel file into {entry: {channel: [...]}}"""
    ev_map: dict[str, dict] = {}
    for suffix in sorted(set(_CHANNEL_FILE.values())):
        path = EVIDENCE_DIR / f"{split}{suffix}.jsonl"
        if not path.exists():
            continue
        with open(path) as fh:
            for ln in fh:
                rec = json.loads(ln)
                ev_map.setdefault(rec["entry"], {}).update(
                    {k: v for k, v in rec.items() if k != "entry"})
    return ev_map


def filter_channels(ev: dict | None, channels: list[str]) -> dict | None:
    if ev is None or channels == ["none"]:
        return None
    keep = ALL_CHANNELS if channels == ["all"] else channels
    out = {k: ev.get(k, []) for k in keep if ev.get(k)}
    return out or None


def cascade_options(item, tree, parent, level, mode, n_max, seed):
    """Build one cascade step's option set (shared by the evaluator and the
    intelligent-search method, so both pose the *same* multiple-choice question).

    Returns ``(options, valid_labels, label2node, gold_labels)`` where options is the
    shuffled candidate list (gold + distractors + UNSURE) and gold_labels is the set
    of option letters that are correct at this level (>=1 for multi-label enzymes).
    """
    true_ecs = item["true_ecs"]
    if mode == "static":
        step = item["chain"][level - 1]
        options, gold_labels = step["options"], set(step["answer_labels"])
    else:
        gold_step = sorted({
            node_at_level(split_ec(e), level)
            for e in true_ecs
            if node_at_level(split_ec(e), level - 1) == parent
        })
        rng = random.Random(f"eval:{item['id']}:{level}:{seed}")
        cands, _ = build_candidates(tree, [parent], gold_step, level, n_max, rng, 0.7)
        labels = _letters(len(cands) + 1)
        options = [{"label": labels[i], "node": cands[i]} for i in range(len(cands))]
        options.append({"label": labels[len(cands)], "node": UNSURE})
        gold_labels = {o["label"] for o in options if o["node"] in set(gold_step)}
    valid_labels = [o["label"] for o in options]
    label2node = {o["label"]: o["node"] for o in options}
    return options, valid_labels, label2node, gold_labels


def run_item(item, tree, backend, mode, ev, n_max, seed, show_seq):
    pred_nodes, stopped, trace = {}, None, []
    parent = "ROOT"
    committed_conf = None  # confidence of the deepest committed level (for calibration/ECE)
    error = None
    try:
        for level in range(1, 5):
            options, valid_labels, label2node, gold_labels = cascade_options(
                item, tree, parent, level, mode, n_max, seed)
            prompt = build_step_prompt(level, parent, options, evidence=ev,
                                       sequence=item["sequence"], seq_features=item.get("seq_features"),
                                       show_sequence=show_seq)
            if hasattr(backend, "set_gold"):
                backend.set_gold(sorted(gold_labels), level)
            label, conf, _raw = backend.act(prompt, valid_labels)
            node = label2node.get(label, UNSURE)
            trace.append({"level": level, "parent": parent, "chosen": node, "conf": conf})
            if node == UNSURE or label is None:
                stopped = level
                break
            pred_nodes[level] = node
            committed_conf = conf
            parent = node
    except Exception as e:  # one bad item (e.g. exhausted API retries) must not kill the pass
        error = str(e)[:200]
    out = {"id": item["id"], "pred_nodes": pred_nodes, "stopped_level": stopped,
           "confidence": committed_conf, "trace": trace}
    if error:
        out["error"] = error
    return out


def run_eval(backend, splits, mode="cascade", channels=None, show_seq=True,
             n_max=10, limit=0, seed=0, workers=1):
    channels = channels or ["none"]
    tree = ECTree.load() if (ITEMS_DIR.parent / "ec_tree.json").exists() else ECTree.from_all_ec()
    # gather (item, evidence) jobs
    jobs = []
    for split in splits:
        items = load_items(split)
        if limit:
            items = subsample(items, limit, split)
        ev_map = load_evidence(split) if channels != ["none"] else {}
        for it in items:
            ev = filter_channels(ev_map.get(it["entry"]), channels) if ev_map else None
            jobs.append((it, ev))

    all_items = [it for it, _ in jobs]
    all_preds = {}
    # Only the (stateless, I/O-bound) API backend is parallel-safe. Mock keeps per-step
    # gold state; HF generation is GIL-bound and not worth threading on one model.
    parallel = workers > 1 and getattr(backend, "name", "").startswith("openai")
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        from ._progress import tracker
        tick = tracker(len(jobs), "cascade")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            preds = ex.map(lambda j: run_item(j[0], tree, backend, mode, j[1], n_max, seed, show_seq), jobs)
            for p in preds:
                all_preds[p["id"]] = p
                tick()
    else:
        for it, ev in jobs:
            all_preds[it["id"]] = run_item(it, tree, backend, mode, ev, n_max, seed, show_seq)
    metrics = evaluate(all_items, all_preds)
    return metrics, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", help="mock | openai:<model> | hf:<model_id>")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--mode", choices=["cascade", "static"], default="cascade")
    ap.add_argument("--channels", nargs="*", default=None,
                    help=f"none | all | subset of {ALL_CHANNELS}")
    ap.add_argument("--open-book", action="store_true", help="alias for --channels all")
    ap.add_argument("--no-seq", action="store_true", help="hide raw sequence (blind ablation)")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel items (API backends only)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ensure_out_dirs()
    channels = args.channels if args.channels else (["all"] if args.open_book else ["none"])
    backend = build_backend(args.backend)
    t0 = time.time()
    metrics, preds = run_eval(backend, args.splits, args.mode, channels, show_seq=not args.no_seq,
                              n_max=args.n_max, limit=args.limit, seed=args.seed, workers=args.workers)
    metrics["meta"] = {"backend": backend.name, "mode": args.mode, "channels": channels,
                       "show_sequence": not args.no_seq, "splits": args.splits, "limit": args.limit,
                       "seconds": round(time.time() - t0, 2)}
    tag = args.tag or f"{backend.name.replace(':', '_')}_{args.mode}_{'+'.join(channels)}{'_noseq' if args.no_seq else ''}"
    (RESULTS_DIR / f"pred_{tag}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds.values()))
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("n", "accuracy", "hierarchical", "abstention", "meta")}, indent=2))


if __name__ == "__main__":
    main()
