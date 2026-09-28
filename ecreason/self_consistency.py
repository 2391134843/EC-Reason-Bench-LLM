"""M4 - Self-Consistency Chain-of-Thought (test-time robustness lever).

A single chain of thought is brittle: it can take a wrong turn early, over-trust a weak
motif, or misread a noisy retrieval hit. Self-Consistency (Wang et al., arXiv:2203.11171)
samples k INDEPENDENT reasoning chains at non-zero temperature and aggregates them by a
vote, keeping the answer the chains agree on and abstaining where they do not.

Here we make it a wrapper over either base reasoner:
  * base="cascade"  -> wraps M1 (hierarchical cascade): k full root->leaf walks.
  * base="mechcot"  -> wraps M3 (mechanism CoT): k single-pass mechanistic EC predictions.

Aggregation is a *prefix vote*: at level L we take the majority node among the chains that
committed at L; if it clears the threshold AND extends the level L-1 consensus, we keep it,
otherwise we stop. The result is the deepest EC prefix the chains agree on -- so the model
commits where it is consistent and abstains where it is not (good calibration). We report the
same hierarchical metrics plus the cost (k model calls) and the realized agreement.

M4 changes the test-time DECISION RULE; it is orthogonal to M3, which changes the reasoning
*content*. Maps to the paper's M4 self-consistency method.

Run:  python -m ecreason.self_consistency --backend mock --base cascade --k 3 --limit 40
      python -m ecreason.self_consistency --backend openai:gpt-4o-mini --base mechcot --k 3  # [API]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from . import motifs as mot
from .baselines import parse_ec
from .build_items import SPLITS
from .ec_tree import ECTree, node_at_level
from .mechanism_cot import build_mechanism_prompt
from .models import build_backend
from .paths import ITEMS_DIR, RESULTS_DIR, ensure_out_dirs
from .run_eval import filter_channels, load_evidence, load_items, run_item, subsample
from .scoring import evaluate


def vote_prefix(chains: list[dict], k: int, threshold: float) -> tuple[dict, dict]:
    """Prefix-vote over k predicted EC paths (each a {level: node} dict).

    Returns (consensus {level: node}, share {level: winning_vote_fraction}). We descend
    level by level: keep the majority node iff its vote share >= threshold and it extends
    the consensus of the previous level; stop at the first level that fails.
    """
    consensus, share = {}, {}
    for level in range(1, 5):
        votes = Counter(c[level] for c in chains if c.get(level))
        if not votes:
            break
        node, cnt = votes.most_common(1)[0]
        frac = cnt / k
        if frac < threshold:
            break
        if level > 1 and node_at_level(node.split("."), level - 1) != consensus.get(level - 1):
            break
        consensus[level] = node
        share[level] = round(frac, 3)
    return consensus, share


def run_self_consistency(backend, splits, base="cascade", channels=None, show_seq=True,
                         k=3, threshold=0.5, temperature=0.7, mode="cascade",
                         n_max=10, limit=0, seed=0, workers=1):
    channels = channels or ["none"]
    tree = ECTree.load() if (ITEMS_DIR.parent / "ec_tree.json").exists() else ECTree.from_all_ec()
    leaves = set(tree.leaves)
    # Diversity across chains comes from sampling: nudge the backend off greedy decoding.
    if hasattr(backend, "temperature") and temperature is not None:
        backend.temperature = temperature

    jobs = []
    for split in splits:
        items = load_items(split)
        if limit:
            items = subsample(items, limit, split)
        ev_map = load_evidence(split) if channels != ["none"] else {}
        for it in items:
            ev = filter_channels(ev_map.get(it["entry"]), channels) if ev_map else None
            jobs.append((it, ev))

    def _chains_cascade(it, ev):
        out = []
        for _ in range(k):  # same seed => same MCQs; diversity from sampling temperature
            r = run_item(it, tree, backend, mode, ev, n_max, seed, show_seq)
            out.append({int(l): n for l, n in r["pred_nodes"].items()})
        return out

    def _chains_mechcot(it, ev):
        motif_hits = mot.scan(it["sequence"])
        prompt = build_mechanism_prompt(it["sequence"], it.get("seq_features"), motif_hits, ev, show_seq)
        out = []
        for _ in range(k):
            if hasattr(backend, "set_zeroshot_gold"):
                backend.set_zeroshot_gold(it["true_ecs"])
            ecs = parse_ec(backend.generate(prompt))
            primary = ecs[0] if ecs else None
            pn = ({lvl: node_at_level(primary.split("."), lvl) for lvl in range(1, 5)}
                  if primary else {})
            out.append(pn)
        return out

    get_chains = _chains_cascade if base == "cascade" else _chains_mechcot

    def _predict(job):
        it, ev = job
        chains = get_chains(it, ev)
        consensus, share = vote_prefix(chains, k, threshold)
        depth = max([0] + list(consensus))
        leaf = consensus.get(4)
        # cascade paths are tree-valid by construction; mechcot consensus may not be.
        valid = True if base == "cascade" else (leaf is None or leaf in leaves)
        return {"id": it["id"], "pred_nodes": consensus, "valid": valid,
                "confidence": share.get(depth), "consensus_depth": depth,
                "vote_share": share, "k": k}

    def _safe(job):
        try:
            return _predict(job)
        except Exception as e:  # a 429/timeout on one enzyme must not kill the whole pass
            return {"id": job[0]["id"], "pred_nodes": {}, "valid": base == "cascade",
                    "confidence": None, "consensus_depth": 0, "vote_share": {}, "k": k,
                    "error": str(e)[:200]}

    all_items = [it for it, _ in jobs]
    all_preds = {}
    parallel = workers > 1 and getattr(backend, "name", "").startswith("openai")
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        from ._progress import tracker
        tick = tracker(len(jobs), f"self-consistency-{base}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(_safe, jobs):
                all_preds[p["id"]] = p
                tick()
    else:
        for job in jobs:
            p = _safe(job)
            all_preds[p["id"]] = p

    metrics = evaluate(all_items, all_preds)
    n = max(1, len(all_preds))
    committed = [p["vote_share"][p["consensus_depth"]] for p in all_preds.values()
                 if p["consensus_depth"] in p["vote_share"]]
    metrics["self_consistency"] = {
        "base": base, "k": k, "threshold": threshold, "temperature": temperature,
        "avg_consensus_depth": round(sum(p["consensus_depth"] for p in all_preds.values()) / n, 3),
        "avg_winning_vote_share": round(sum(committed) / max(1, len(committed)), 3),
        "model_calls_per_enzyme": k if base == "mechcot" else f"~{k}x cascade walk",
    }
    return metrics, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", help="mock | openai:<model> | hf:<model_id>")
    ap.add_argument("--base", choices=["cascade", "mechcot"], default="cascade",
                    help="reasoner to self-consistency-wrap: cascade=M1, mechcot=M3")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--channels", nargs="*", default=["none"], help="none (closed-book) | all | subset")
    ap.add_argument("--k", type=int, default=3, help="number of sampled chains")
    ap.add_argument("--threshold", type=float, default=0.5, help="min vote share to commit a level")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for chains")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    t0 = time.time()
    metrics, preds = run_self_consistency(
        backend, args.splits, base=args.base, channels=args.channels, show_seq=not args.no_seq,
        k=args.k, threshold=args.threshold, temperature=args.temperature,
        n_max=args.n_max, limit=args.limit, seed=args.seed, workers=args.workers)
    metrics["meta"] = {"method": f"self-consistency-{args.base}", "backend": backend.name,
                       "channels": args.channels, "show_sequence": not args.no_seq,
                       "splits": args.splits, "limit": args.limit, "seconds": round(time.time() - t0, 2)}
    tag = args.tag or f"sc_{args.base}_{backend.name.replace(':', '_')}_{'+'.join(args.channels)}"
    (RESULTS_DIR / f"pred_{tag}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds.values()))
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in
                      ("n", "accuracy", "valid_ec_rate", "self_consistency", "meta")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
