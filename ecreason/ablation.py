"""Ablation runner: treat each information setting as a separate experiment.

Each row below is one condition (a different amount of sequence info / evidence). We
run the cascade eval under each and tabulate per-level accuracy + hierarchical F1, so
the contribution of every channel is isolated (plan §8 / RQ1: format vs knowledge).

NOTE: with --backend mock the conditions look similar (mock ignores the prompt); the
ablation is meaningful with a real backend (openai:* [API] or hf:* [GPU]).

Run:  python -m ecreason.ablation --backend mock --limit 80
      python -m ecreason.ablation --backend openai:gpt-4o-mini      # [API]
      python -m ecreason.ablation --backend hf:Qwen/Qwen2.5-7B-Instruct  # [GPU]
"""
from __future__ import annotations

import argparse
import json

from .build_items import SPLITS
from .models import build_backend
from .paths import RESULTS_DIR, ensure_out_dirs
from .run_eval import run_eval

# (name, channels, show_sequence)
ABLATIONS = [
    ("blind (no seq, no evidence)", ["none"], False),
    ("closed-book (seq + features)", ["none"], True),
    ("+ ESM-kNN", ["esm_knn"], True),
    ("+ BLAST (DIAMOND)", ["blast"], True),
    ("+ HMMER (phmmer)", ["hmmer"], True),
    ("+ structure (Foldseek)", ["structure"], True),
    ("+ active-site (Folddisco)", ["active_site"], True),
    ("open-book (all channels)", ["all"], True),
]


def _md_table(rows: list[dict]) -> str:
    head = "| condition | L1 | L2 | L3 | L4 | H-F1 | avg_commit |\n|---|--:|--:|--:|--:|--:|--:|\n"
    body = ""
    for r in rows:
        a = r["accuracy"]
        body += (f"| {r['name']} | {a.get('L1','-')} | {a.get('L2','-')} | {a.get('L3','-')} | "
                 f"{a.get('L4','-')} | {r['h_f1']} | {r['avg_commit']} |\n")
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--mode", choices=["cascade", "static"], default="cascade")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="parallel items (API backends only)")
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    rows = []
    for name, channels, show_seq in ABLATIONS:
        metrics, _ = run_eval(backend, args.splits, args.mode, channels,
                              show_seq=show_seq, limit=args.limit, seed=args.seed, workers=args.workers)
        rows.append({
            "name": name, "channels": channels, "show_seq": show_seq,
            "accuracy": metrics["accuracy"],
            "h_f1": metrics["hierarchical"].get("f1"),
            "avg_commit": metrics["abstention"].get("avg_commit_level"),
            "n": metrics["n"],
        })
        print(f"[ablation] {name}: {json.dumps(metrics['accuracy'])}", flush=True)

    tag = backend.name.replace(":", "_")
    out = {"backend": backend.name, "mode": args.mode, "splits": args.splits,
           "limit": args.limit, "rows": rows}
    (RESULTS_DIR / f"ablation_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    md = f"# Ablation — backend={backend.name}, mode={args.mode}, n/cond≈{rows[0]['n']}\n\n" + _md_table(rows)
    (RESULTS_DIR / f"ablation_{tag}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
