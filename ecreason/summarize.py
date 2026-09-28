"""Aggregate a multi-condition run into one comparable report.

Each condition was run as two full passes (the two homology-graded splits followed by
the adversarial and multi-label splits), writing
``pred_<prefix><cond>_big.jsonl`` / ``_small.jsonl``. This script
unions the predictions per condition and RE-SCORES them over all four splits in one call,
so the overall numbers, per-split breakdowns and the extra metrics (exact-EC, macro, ECE)
are all exact. Method-specific diagnostics (M2 search cost, M4 self-consistency) are read
back from the per-pass metrics files and merged.

Run:  python -m ecreason.summarize                      # default prefix 'ds_'
      python -m ecreason.summarize --prefix ds_ --out summary_deepseek
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .build_items import SPLITS
from .paths import RESULTS_DIR, ensure_out_dirs
from .run_eval import load_items
from .scoring import evaluate

# (display name, condition tag) — must match the orchestration runner.
CONDITIONS = [
    ("B0 zero-shot, closed-book (CARE/ChatGPT)", "b0_closed"),
    ("B0 zero-shot, open-book (reference)", "b0_open"),
    ("M1 cascade, closed-book", "m1"),
    ("M2 intelligent search, active retrieval", "m2"),
    ("Cascade, open-book, all evidence (reference)", "casc_open"),
    ("M3 mechanistic CoT, closed-book", "m3"),
    ("M3 mechanistic CoT, open-book", "m3_open"),
    ("M4 self-consistency, k=3, closed-book", "m4"),
    ("M4 self-consistency, k=3, open-book", "m4_open"),
]


def _load_preds(prefix: str, cond: str) -> dict:
    preds = {}
    for pass_ in ("big", "small"):
        path = RESULTS_DIR / f"pred_{prefix}{cond}_{pass_}.jsonl"
        if not path.exists():
            continue
        for ln in path.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                preds[rec["id"]] = rec
    return preds


def _load_metrics(prefix: str, cond: str) -> list[dict]:
    out = []
    for pass_ in ("big", "small"):
        path = RESULTS_DIR / f"metrics_{prefix}{cond}_{pass_}.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out


def _merge_search_cost(preds: dict, diag: list[dict]) -> dict | None:
    """Recompute M2 search cost over the union of both passes.

    Averaging the per-pass blocks is wrong: each pass divides by its own sample
    count, so keeping only the first one reports the cost of that pass alone.
    Every channel is fetched at most once per enzyme, hence one tool call per
    search, and the columns sum to `avg_searches_per_enzyme` times `n`.
    """
    if not any("n_search" in p for p in preds.values()):
        return None
    n = max(1, len(preds))
    tool_use = Counter(t for p in preds.values() for t in p.get("tools", []))
    order = ["blast", "hmmer", "esm_knn", "structure", "active_site", "pfam"]
    tools = order + [t for t in tool_use if t not in order]
    budgets = [d["search_cost"].get("max_search_budget") for d in diag if d.get("search_cost")]
    return {
        "avg_searches_per_enzyme": round(sum(p.get("n_search", 0) for p in preds.values()) / n, 3),
        "avg_model_turns_per_enzyme": round(sum(p.get("n_turns", 0) for p in preds.values()) / n, 3),
        "tool_usage": {t: tool_use.get(t, 0) for t in tools},
        "max_search_budget": max([b for b in budgets if b is not None], default=None),
        "n_enzymes": n,
    }


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ds_")
    ap.add_argument("--out", default="summary_deepseek")
    ap.add_argument("--title", default=None, help="model label for the report title")
    args = ap.parse_args()
    ensure_out_dirs()

    all_items = {s: load_items(s) for s in SPLITS}
    rows, results = [], {}
    for name, cond in CONDITIONS:
        preds = _load_preds(args.prefix, cond)
        if not preds:
            continue
        items = [it for s in SPLITS for it in all_items[s] if it["id"] in preds]
        m = evaluate(items, preds)
        diag = _load_metrics(args.prefix, cond)
        # merge method diagnostics across the two passes (sum / weighted)
        search_cost = _merge_search_cost(preds, diag)
        sc = next((d.get("self_consistency") for d in diag if d.get("self_consistency")), None)
        results[cond] = {"name": name, "metrics": m, "search_cost": search_cost, "self_consistency": sc}
        rows.append((name, cond, m))

    a = lambda m, k: m["accuracy"].get(k)
    title = args.title or f"prefix={args.prefix}"
    lines = [f"# {title}: EC-Reason-Bench result summary", "",
             f"> {len(rows)} conditions; every condition is evaluated on all 1,349 items.", ""]
    lines += ["## Overall results", "",
              "| Condition | n | L1 | L2 | L3 | L4 | exact-EC | H-F1 | valid-EC | macro-L4 | ECE | avg_commit |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, cond, m in rows:
        lines.append(
            f"| {name} | {m['n']} | {_fmt(a(m,'L1'))} | {_fmt(a(m,'L2'))} | {_fmt(a(m,'L3'))} | "
            f"{_fmt(a(m,'L4'))} | {_fmt(m['exact_ec_match'])} | {_fmt(m['hierarchical'].get('f1'))} | "
            f"{_fmt(m['valid_ec_rate'])} | {_fmt(m['macro_accuracy'].get('L4'))} | "
            f"{_fmt(m['calibration'].get('ece'))} | {_fmt(m['abstention'].get('avg_commit_level'),3)} |")

    for s in SPLITS:
        lines += ["", f"## Split: {s}", "", "| Condition | n | L1 | L2 | L3 | L4 |",
                  "|---|--:|--:|--:|--:|--:|"]
        for name, cond, m in rows:
            bs = m["by_split"].get(s, {})
            lines.append(f"| {name} | {bs.get('n','-')} | {_fmt(bs.get('L1'))} | {_fmt(bs.get('L2'))} | "
                         f"{_fmt(bs.get('L3'))} | {_fmt(bs.get('L4'))} |")

    # method-specific diagnostics
    lines += ["", "## Method-specific diagnostics", ""]
    for name, cond, m in rows:
        d = results[cond]
        if d["search_cost"]:
            sc = d["search_cost"]
            lines.append(
                f"- **{name}** retrieval cost: {sc.get('avg_searches_per_enzyme')} "
                f"calls/enzyme and {sc.get('avg_model_turns_per_enzyme')} model turns/enzyme; "
                f"tool use: {sc.get('tool_usage')}"
            )
        if d["self_consistency"]:
            s4 = d["self_consistency"]
            lines.append(
                f"- **{name}** self-consistency: base={s4.get('base')}, k={s4.get('k')}; "
                f"mean consensus depth={s4.get('avg_consensus_depth')}, "
                f"mean winning vote share={s4.get('avg_winning_vote_share')}"
            )

    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"{args.out}.md").write_text(md)
    (RESULTS_DIR / f"{args.out}.json").write_text(
        json.dumps({c: {"name": r["name"], "metrics": r["metrics"],
                        "search_cost": r["search_cost"], "self_consistency": r["self_consistency"]}
                    for c, r in results.items()}, indent=2, ensure_ascii=False))
    print(md)
    print(f"\n[summarize] wrote {RESULTS_DIR / (args.out + '.md')}")


if __name__ == "__main__":
    main()
