"""Generate cascading multiple-choice items for every test split (offline).

Each item carries the gold EC path plus, for every level, a precomputed candidate
set (gold option(s) + hard sibling distractors + easy distractors + UNSURE). The
candidate sets enable *static* (teacher-forced-parent) evaluation and inspection;
the *cascade* evaluator regenerates candidates dynamically from the EC tree using
the same `build_candidates` routine (see run_eval.py).

Run:  python -m ecreason.build_items --n-max 10 --hard-ratio 0.7 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random

from . import difficulty as diff
from . import seq_features as sf
from .data_utils import load_split
from .ec_tree import ECTree, UNSURE, build_candidates, node_at_level, split_ec
from .identifiers import public_query_id
from .paths import ITEMS_DIR, ensure_out_dirs

SPLITS = ["30", "30-50", "price", "promiscuous"]


def _letters(n: int) -> list[str]:
    # A..Z then AA.. (more than enough for n_max+1)
    out = []
    for i in range(n):
        if i < 26:
            out.append(chr(ord("A") + i))
        else:
            out.append(chr(ord("A") + i // 26 - 1) + chr(ord("A") + i % 26))
    return out


def build_item(tree: ECTree, split: str, row: dict, n_max: int, hard_ratio: float, seed: int) -> dict:
    entry, ecs = row["entry"], row["ecs"]
    valid_ecs = [e for e in ecs if len(split_ec(e)) == 4]
    if not valid_ecs:
        return None
    rng = random.Random(f"{split}:{entry}:{seed}")

    chain = []
    branches = []
    hard_shares = []
    for level in range(1, 5):
        gold_parents = {node_at_level(split_ec(e), level - 1) for e in valid_ecs}
        gold_opts = sorted({node_at_level(split_ec(e), level) for e in valid_ecs})
        cands, hard_share = build_candidates(
            tree, gold_parents, gold_opts, level, n_max, rng, hard_ratio
        )
        # attach option letters; UNSURE always last
        labels = _letters(len(cands) + 1)
        options = [{"label": labels[i], "node": cands[i]} for i in range(len(cands))]
        options.append({"label": labels[len(cands)], "node": UNSURE})
        gold_labels = [o["label"] for o in options if o["node"] in set(gold_opts)]
        chain.append({
            "level": level,
            "parents": sorted(gold_parents),
            "options": options,
            "answer_nodes": gold_opts,
            "answer_labels": gold_labels,
        })
        branches.append(sum(len(tree.get_children(p)) for p in gold_parents) / max(1, len(gold_parents)))
        hard_shares.append(hard_share)

    avg_branch = sum(branches) / len(branches)
    distractor_hardness = sum(hard_shares) / len(hard_shares)
    dscore = diff.composite(split, valid_ecs, avg_branch, distractor_hardness)

    public_entry = public_query_id(split, entry)
    return {
        "id": f"{split}::{public_entry}",
        "split": split,
        "entry": public_entry,
        "sequence": row["sequence"],
        "seq_features": sf.summarize(row["sequence"]),  # intrinsic, closed-book features
        "true_ecs": valid_ecs,
        "n_labels": len(valid_ecs),
        "chain": chain,
        "difficulty": {**dscore, "avg_branch": round(avg_branch, 2)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=10, help="max options per level (excl. UNSURE)")
    ap.add_argument("--hard-ratio", type=float, default=0.7, help="share of hard (sibling) distractors")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_out_dirs()
    tree = ECTree.from_all_ec()
    tree.save()

    all_items = []
    for split in SPLITS:
        for row in load_split(split):
            it = build_item(tree, split, row, args.n_max, args.hard_ratio, args.seed)
            if it is not None:
                all_items.append(it)

    # global difficulty binning across all splits
    bins = diff.quantile_bins([it["difficulty"]["d"] for it in all_items])
    for it, b in zip(all_items, bins):
        it["difficulty"]["bin"] = b

    # write one jsonl per split + a combined manifest
    per_split = {s: [] for s in SPLITS}
    for it in all_items:
        per_split[it["split"]].append(it)
    summary = {}
    for split, items in per_split.items():
        out = ITEMS_DIR / f"{split}.jsonl"
        with open(out, "w") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        bin_counts = {}
        for it in items:
            bin_counts[it["difficulty"]["bin"]] = bin_counts.get(it["difficulty"]["bin"], 0) + 1
        summary[split] = {
            "n_items": len(items),
            "n_multilabel": sum(1 for it in items if it["n_labels"] > 1),
            "difficulty_bins": dict(sorted(bin_counts.items())),
        }
    with open(ITEMS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
