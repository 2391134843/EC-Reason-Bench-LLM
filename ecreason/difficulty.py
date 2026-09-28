"""Per-item difficulty scoring along orthogonal axes (see plan §5).

All axes are normalized to [0, 1] (higher = harder). Because every EC in this
dataset is a full 4-level leaf, the "depth" axis is constant at the item level and
is therefore reported separately (per-level analysis) rather than folded into the
composite. The composite uses: split-hardness, tail (rarity), branching, and
realized distractor-hardness.
"""
from __future__ import annotations

from .data_utils import tail_score

# Low homology to train => hardest. Mirrors CARE split semantics.
SPLIT_HARDNESS = {
    "30": 1.00,        # <30% identity: stringent low-homology
    "30-50": 0.50,     # intermediate homology
    "price": 0.80,     # previously misclassified (adversarial)
    "promiscuous": 0.70,  # multi-label
}

_BRANCH_NORM = 30.0  # branching factors are clipped/normalized by this


def branch_score(avg_branch: float) -> float:
    return min(1.0, avg_branch / _BRANCH_NORM)


def composite(split: str, ecs: list[str], avg_branch: float, distractor_hardness: float) -> dict:
    axes = {
        "split_hardness": SPLIT_HARDNESS.get(split, 0.7),
        "tail": tail_score(ecs),
        "branch": branch_score(avg_branch),
        "distractor_hardness": distractor_hardness,
    }
    d = sum(axes.values()) / len(axes)
    return {"d": round(d, 4), "axes": {k: round(v, 4) for k, v in axes.items()}}


def quantile_bins(scores: list[float], n_bins: int = 5) -> list[str]:
    """Assign D1..D5 by global quantiles. Stable for ties."""
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    bins = [""] * len(scores)
    n = len(scores)
    for rank, idx in enumerate(order):
        b = min(n_bins - 1, int(rank * n_bins / n))
        bins[idx] = f"D{b + 1}"
    return bins
