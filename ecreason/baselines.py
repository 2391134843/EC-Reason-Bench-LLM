"""Zero-shot DIRECT EC-number prediction baseline (reproduces the CARE/ChatGPT setting).

This is the zero-shot baseline used for comparison with the cascade method. The model sees the
query (sequence [+ features] [+ evidence]) and must emit the EC number in ONE shot as
free text — no hierarchy, no candidate list, no abstention. We then:
  * parse the EC number(s) from the free text,
  * score with the SAME hierarchical metrics as the cascade (so it's comparable),
  * report `valid_ec_rate` = fraction of outputs that are REAL EC numbers (the rest are
    hallucinated / non-existent) — the headline failure mode our cascade removes by design.

Conditions (set via --channels and --no-seq, same knobs as run_eval):
  zero-shot closed-book : --channels none           (sequence + features only)
  zero-shot open-book   : --channels all            (direct gen, but WITH retrieval evidence)
  blind                 : --channels none --no-seq  (no sequence at all)

Run:  python -m ecreason.baselines --backend mock --limit 40
      python -m ecreason.baselines --backend openai:gpt-4o-mini --channels none   # [API]
"""
from __future__ import annotations

import argparse
import json
import re
import time

from .build_items import SPLITS
from .ec_tree import ECTree, node_at_level
from .models import build_backend
from .paths import ITEMS_DIR, RESULTS_DIR, ensure_out_dirs
from .prompts import render_retrieval, render_seq_features, render_sequence
from .run_eval import filter_channels, load_evidence, load_items, subsample
from .scoring import evaluate

_EC_FULL = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d)")

ZS_SYSTEM = (
    "You are an expert enzyme annotator. Predict the Enzyme Commission (EC) number of the "
    "query enzyme. Think briefly, then give the answer on the final line as:\n"
    "    EC=a.b.c.d\n"
    "Use the full 4-field EC number. If the enzyme is promiscuous, you may list several "
    "separated by ';'. Do not invent fields you are unsure of."
)


def build_zeroshot_prompt(sequence, seq_features=None, evidence=None, show_sequence=True) -> str:
    seq_block = f"\n[Query sequence]\n{render_sequence(sequence)}\n" if show_sequence else ""
    ev_block = f"\n[Retrieval evidence]\n{render_retrieval(evidence)}\n" if evidence else ""
    return (
        f"{ZS_SYSTEM}\n"
        f"{seq_block}"
        f"\n[Sequence-derived features]\n{render_seq_features(seq_features, sequence)}\n"
        f"{ev_block}"
        f"\n[Answer] Reason in 1-3 sentences, then a final line 'EC=a.b.c.d'."
    )


def parse_ec(text: str) -> list[str]:
    """Extract full 4-field EC numbers from free text (in order, de-duplicated)."""
    out = []
    for m in _EC_FULL.findall(text or ""):
        ec = ".".join(m)
        if ec not in out:
            out.append(ec)
    return out


def run_zeroshot(backend, splits, channels=None, show_seq=True, limit=0, seed=0, workers=1):
    channels = channels or ["none"]
    tree = ECTree.load() if (ITEMS_DIR.parent / "ec_tree.json").exists() else ECTree.from_all_ec()
    leaves = set(tree.leaves)

    jobs = []
    for split in splits:
        items = load_items(split)
        if limit:
            items = subsample(items, limit, split)
        ev_map = load_evidence(split) if channels != ["none"] else {}
        for it in items:
            ev = filter_channels(ev_map.get(it["entry"]), channels) if ev_map else None
            jobs.append((it, ev))

    def _predict(job):
        it, ev = job
        prompt = build_zeroshot_prompt(it["sequence"], it.get("seq_features"), ev, show_seq)
        if hasattr(backend, "set_zeroshot_gold"):
            backend.set_zeroshot_gold(it["true_ecs"])
        try:
            raw = backend.generate(prompt)
        except Exception as e:  # one bad item must not kill the pass
            return {"id": it["id"], "pred_nodes": {}, "pred_ecs": [], "valid": False,
                    "raw": "", "error": str(e)[:200]}
        ecs = parse_ec(raw)
        primary = ecs[0] if ecs else None
        pred_nodes = {}
        if primary:
            parts = primary.split(".")
            pred_nodes = {lvl: node_at_level(parts, lvl) for lvl in range(1, 5)}
        return {"id": it["id"], "pred_nodes": pred_nodes, "pred_ecs": ecs,
                "valid": bool(primary in leaves), "raw": (raw or "")[:200]}

    all_items = [it for it, _ in jobs]
    all_preds = {}
    parallel = workers > 1 and getattr(backend, "name", "").startswith("openai")
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        from ._progress import tracker
        tick = tracker(len(jobs), "zero-shot")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(_predict, jobs):
                all_preds[p["id"]] = p
                tick()
    else:
        for job in jobs:
            p = _predict(job)
            all_preds[p["id"]] = p
    metrics = evaluate(all_items, all_preds)
    return metrics, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--channels", nargs="*", default=["none"], help="none | all | subset")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    t0 = time.time()
    metrics, preds = run_zeroshot(backend, args.splits, args.channels, show_seq=not args.no_seq,
                                  limit=args.limit, seed=args.seed, workers=args.workers)
    metrics["meta"] = {"method": "zero-shot-direct", "backend": backend.name,
                       "channels": args.channels, "show_sequence": not args.no_seq,
                       "splits": args.splits, "limit": args.limit, "seconds": round(time.time() - t0, 2)}
    tag = args.tag or f"zeroshot_{backend.name.replace(':', '_')}_{'+'.join(args.channels)}"
    (RESULTS_DIR / f"pred_{tag}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds.values()))
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("n", "accuracy", "valid_ec_rate", "hierarchical", "meta")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
