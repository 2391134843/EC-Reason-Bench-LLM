"""Head-to-head: zero-shot DIRECT prediction (baseline) vs OUR methods, one comparable table.

Runs the same model under every condition so the contribution of each lever is isolated:

  1. zero-shot, closed-book        - the CARE/ChatGPT setting (direct EC generation)        [B0]
  2. zero-shot, open-book          - direct generation but WITH retrieval evidence
  3. OURS M3: mechanism-CoT, closed-book - biochemically-structured reasoning, NO tree, NO retrieval
  4. OURS M1: cascade, closed-book - hierarchy-guided MCQ, sequence + features
  5. OURS M2: intelligent search   - agentic ReAct retrieval over the cascade (model picks tools)
  6. cascade, open-book (all)      - non-agentic upper-reference: every evidence channel dumped in
  (+ --with-sc) OURS M4: self-consistency over the cascade (k-sample prefix vote)

Levers, each isolated against the same baseline:
  (row3 vs row1) = *reasoning* structure (mechanism schema, closed-book);
  (row4 vs row1) = *output* structure (EC tree cascade);
  (row5 vs row4) = *external knowledge* via active search; (row6) = same evidence, dumped passively;
  (M4)           = test-time *robustness* (sampling + vote).

Headline columns: per-level accuracy (ori_acc), hierarchical F1, and valid-EC rate
(free-form generation hallucinates non-existent EC numbers; the cascade family is 100% by
construction; mechanism-CoT is free-form so its valid-EC rate is diagnostic too).

Run:  python -m ecreason.compare --backend mock --limit 60
      python -m ecreason.compare --backend openai:gpt-4o-mini --workers 8 --with-sc   # [API]
"""
from __future__ import annotations

import argparse
import json

from .baselines import run_zeroshot
from .build_items import SPLITS
from .mechanism_cot import run_mechanism_cot
from .models import build_backend
from .paths import RESULTS_DIR, ensure_out_dirs
from .run_eval import run_eval
from .search import run_search
from .self_consistency import run_self_consistency

# (display name, method, channels, show_seq)
CONDITIONS = [
    ("zero-shot · closed-book (CARE/ChatGPT setting)", "zeroshot", ["none"], True),
    ("zero-shot · open-book (direct + evidence)",      "zeroshot", ["all"], True),
    ("OURS M3 mechanism-CoT · closed-book",            "mechcot",  ["none"], True),
    ("OURS M1 cascade · closed-book",                  "cascade",  ["none"], True),
    ("OURS M2 intelligent-search · active retrieval",  "search",   ["all"], True),
    ("OURS M1 cascade · open-book (all evidence)",     "cascade",  ["all"], True),
]


def _row(name, method, metrics):
    a = metrics["accuracy"]
    return {
        "condition": name, "method": method,
        "L1": a.get("L1"), "L2": a.get("L2"), "L3": a.get("L3"), "L4": a.get("L4"),
        "h_f1": metrics["hierarchical"].get("f1"),
        "valid_ec_rate": metrics.get("valid_ec_rate"),
        "avg_commit": metrics["abstention"].get("avg_commit_level"),
        "n": metrics["n"],
    }


def _md(rows) -> str:
    h = ("| Method / condition | L1 | L2 | L3 | L4 | H-F1 | valid-EC | avg_commit |\n"
         "|---|--:|--:|--:|--:|--:|--:|--:|\n")
    b = ""
    for r in rows:
        b += (f"| {r['condition']} | {r['L1']} | {r['L2']} | {r['L3']} | {r['L4']} "
              f"| {r['h_f1']} | {r['valid_ec_rate']} | {r['avg_commit']} |\n")
    return h + b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--with-sc", action="store_true",
                    help="also run M4 self-consistency (k-sample vote; ~k x model calls)")
    ap.add_argument("--sc-k", type=int, default=3)
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    conditions = list(CONDITIONS)
    if args.with_sc:
        conditions.append(
            (f"OURS M4 self-consistency · cascade (k={args.sc_k})", "sc", ["none"], True))

    rows = []
    for name, method, channels, show_seq in conditions:
        if method == "zeroshot":
            metrics, _ = run_zeroshot(backend, args.splits, channels, show_seq=show_seq,
                                      limit=args.limit, seed=args.seed, workers=args.workers)
        elif method == "mechcot":
            metrics, _ = run_mechanism_cot(backend, args.splits, channels, show_seq=show_seq,
                                           limit=args.limit, seed=args.seed, workers=args.workers)
        elif method == "search":
            metrics, _ = run_search(backend, args.splits, channels=channels, show_seq=show_seq,
                                    limit=args.limit, seed=args.seed, workers=args.workers)
        elif method == "sc":
            metrics, _ = run_self_consistency(backend, args.splits, base="cascade", channels=channels,
                                              show_seq=show_seq, k=args.sc_k, limit=args.limit,
                                              seed=args.seed, workers=args.workers)
        else:
            metrics, _ = run_eval(backend, args.splits, "cascade", channels, show_seq=show_seq,
                                  limit=args.limit, seed=args.seed, workers=args.workers)
        rows.append(_row(name, method, metrics))
        print(f"[compare] {name}: L4={rows[-1]['L4']} valid-EC={rows[-1]['valid_ec_rate']}", flush=True)

    tag = backend.name.replace(":", "_")
    out = {"backend": backend.name, "splits": args.splits, "limit": args.limit, "rows": rows}
    (RESULTS_DIR / f"compare_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    md = (f"# Zero-shot baseline vs. diagnostic methods — backend={backend.name}, "
          f"n/condition={rows[0]['n']}\n\n"
          + _md(rows)
          + "\n> valid-EC is the fraction of predictions that are real EC numbers. "
          "B0 and M3 use free generation, so valid-EC is diagnostic. M3 is closed-book, "
          "does not traverse the tree, and organizes reasoning around four biochemical "
          "axes plus intrinsic sequence motifs. M1 and M2 draw every candidate from the "
          "EC tree, so valid-EC is structurally 1.0. M2 actively selects retrieval tools "
          "inside a ReAct-style loop, unlike passive open-book prompting.\n")
    (RESULTS_DIR / f"compare_{tag}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
