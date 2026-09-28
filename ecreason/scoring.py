"""Hierarchical metrics aligned with the CARE `ori_acc` philosophy (plan §8).

A level-L prediction is correct iff the model committed a node at level L (i.e. did
not abstain before reaching L) and that node matches the level-L prefix of *some*
true EC (handles multi-label). We also report hierarchical micro-P/R/F1, abstention
behavior, and a per-difficulty-bin breakdown.
"""
from __future__ import annotations

from collections import defaultdict

from .ec_tree import node_at_level, split_ec


def gold_prefixes(true_ecs: list[str], level: int) -> set[str]:
    return {node_at_level(split_ec(e), level) for e in true_ecs}


def gold_ancestor_set(true_ecs: list[str]) -> set[str]:
    s = set()
    for e in true_ecs:
        parts = split_ec(e)
        for lvl in range(1, 5):
            s.add(node_at_level(parts, lvl))
    return s


def score_item(true_ecs: list[str], pred_nodes: dict[int, str]) -> dict:
    """pred_nodes: {level: node} for committed levels only (1..4)."""
    per_level = {}
    for lvl in range(1, 5):
        node = pred_nodes.get(lvl)
        per_level[lvl] = bool(node is not None and node in gold_prefixes(true_ecs, lvl))
    pred_set = set(pred_nodes.values())
    gold_set = gold_ancestor_set(true_ecs)
    inter = len(pred_set & gold_set)
    prec = inter / len(pred_set) if pred_set else 0.0
    rec = inter / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"per_level": per_level, "h_prec": prec, "h_rec": rec, "h_f1": f1,
            "committed_to": max([0] + [l for l in pred_nodes])}


def _ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float | None:
    """Expected Calibration Error over (confidence, correct) pairs (equal-width bins)."""
    if not pairs:
        return None
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        b = min(n_bins - 1, max(0, int(conf * n_bins)))
        bins[b].append((conf, correct))
    n = len(pairs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(x[1] for x in b) / len(b)
        conf = sum(x[0] for x in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)
    return round(ece, 4)


def evaluate(items: list[dict], predictions: dict[str, dict]) -> dict:
    """items: list of benchmark items. predictions: {id: {"pred_nodes": {level:node}, ...}}.

    Reports (plan §8): per-level ori_acc (L1-L4), hierarchical micro P/R/F1, valid-EC rate,
    abstention/commit-depth, per-split and per-difficulty breakdowns, plus three extra metrics:
      * exact_ec_match            - fraction with the full 4-field EC exactly correct (== L4 acc).
      * full_ec_precision         - of items that committed all the way to L4, how many are right.
      * macro_accuracy            - per-level accuracy macro-averaged over the 7 EC main classes.
      * calibration.ece           - ECE over predictions that carry a `confidence` field.
    """
    level_correct = {l: 0 for l in range(1, 5)}
    hp = hr = hf = 0.0
    n = 0
    n_valid = 0
    reached_l4 = 0
    correct_l4_committed = 0
    stop_levels = []
    conf_pairs: list[tuple[float, int]] = []
    by_bin = defaultdict(lambda: {l: 0 for l in range(1, 5)} | {"n": 0})
    by_split = defaultdict(lambda: {l: 0 for l in range(1, 5)} | {"n": 0})
    by_class = defaultdict(lambda: {l: 0 for l in range(1, 5)} | {"n": 0})

    for it in items:
        pred = predictions.get(it["id"], {})
        pred_nodes = {int(k): v for k, v in pred.get("pred_nodes", {}).items()}
        s = score_item(it["true_ecs"], pred_nodes)
        n += 1
        for l in range(1, 5):
            level_correct[l] += int(s["per_level"][l])
        hp += s["h_prec"]; hr += s["h_rec"]; hf += s["h_f1"]
        # valid-EC: cascade preds omit "valid" (candidates are always tree-valid -> True);
        # the zero-shot baseline sets it explicitly (False when it emits a non-existent EC).
        n_valid += int(pred.get("valid", True))
        committed = s["committed_to"]
        stop_levels.append(committed)
        if committed == 4:
            reached_l4 += 1
            correct_l4_committed += int(s["per_level"][4])
        # calibration: only for methods that emit a confidence (cascade / search / self-consistency)
        conf = pred.get("confidence")
        if conf is not None and committed >= 1:
            conf_pairs.append((float(conf), int(s["per_level"][committed])))
        b = it["difficulty"]["bin"]
        by_bin[b]["n"] += 1
        by_split[it["split"]]["n"] += 1
        for cls in gold_prefixes(it["true_ecs"], 1):  # macro over EC main classes (multi-label -> each)
            by_class[cls]["n"] += 1
        for l in range(1, 5):
            by_bin[b][l] += int(s["per_level"][l])
            by_split[it["split"]][l] += int(s["per_level"][l])
            for cls in gold_prefixes(it["true_ecs"], 1):
                by_class[cls][l] += int(s["per_level"][l])

    def _acc(counts, n_):
        return {f"L{l}": round(counts[l] / n_, 4) for l in range(1, 5)} if n_ else {}

    macro = ({f"L{l}": round(sum(v[l] / v["n"] for v in by_class.values()) / len(by_class), 4)
              for l in range(1, 5)} if by_class else {})

    return {
        "n": n,
        "accuracy": _acc(level_correct, n),
        "hierarchical": {"precision": round(hp / n, 4), "recall": round(hr / n, 4), "f1": round(hf / n, 4)} if n else {},
        "abstention": {"avg_commit_level": round(sum(stop_levels) / n, 3) if n else 0.0,
                       "reached_L4": round(sum(1 for s in stop_levels if s == 4) / n, 4) if n else 0.0},
        "by_difficulty": {b: {**_acc(v, v["n"]), "n": v["n"]} for b, v in sorted(by_bin.items())},
        "by_split": {s: {**_acc(v, v["n"]), "n": v["n"]} for s, v in by_split.items()},
        # fraction of predictions that are a REAL EC. 1.0 for cascade (tree-constrained);
        # < 1.0 for zero-shot direct generation that hallucinates non-existent EC numbers.
        "valid_ec_rate": round(n_valid / n, 4) if n else 1.0,
        # --- extra metrics (per user request) ---
        "exact_ec_match": round(level_correct[4] / n, 4) if n else 0.0,
        "full_ec_precision": round(correct_l4_committed / reached_l4, 4) if reached_l4 else None,
        "macro_accuracy": {**macro, "n_classes": len(by_class)},
        "calibration": {"ece": _ece(conf_pairs), "n_with_conf": len(conf_pairs)},
    }
