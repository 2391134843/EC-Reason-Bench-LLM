"""EC hierarchy as a tree, plus a shared candidate-set builder.

An EC number ``a.b.c.d`` induces a path  ROOT -> a.-.-.- -> a.b.-.- -> a.b.c.- -> a.b.c.d.
We represent internal/partial nodes with dashes, e.g. ``"1.8.-.-"`` (level 2).

The same `build_candidates` function is used both when *generating* the static
benchmark items (parent = gold parent) and when *running* the cascade evaluation
(parent = the node the model actually chose), guaranteeing consistent option sets.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Iterable

from .paths import ALL_EC_TXT, EC_TREE_JSON

UNSURE = "UNSURE"  # explicit abstain/stop option offered at every level


# --------------------------------------------------------------------------- #
# Prefix helpers
# --------------------------------------------------------------------------- #
def split_ec(ec: str) -> list[str]:
    return ec.strip().split(".")


def node_at_level(parts: list[str], level: int) -> str:
    """Return the EC node string truncated to `level` (0 -> ROOT)."""
    if level <= 0:
        return "ROOT"
    p = list(parts[:level]) + ["-"] * (4 - level)
    return ".".join(p)


def node_level(node: str) -> int:
    if node == "ROOT":
        return 0
    return sum(1 for x in node.split(".") if x != "-")


def parent_of(node: str) -> str:
    lvl = node_level(node)
    if lvl <= 1:
        return "ROOT"
    return node_at_level(node.split("."), lvl - 1)


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
class ECTree:
    def __init__(self, leaves: Iterable[str]):
        self.children: dict[str, list[str]] = {}
        self.nodes_by_level: dict[int, set[str]] = defaultdict(set)
        _children: dict[str, set[str]] = defaultdict(set)
        self.leaves: list[str] = sorted(set(leaves))
        for leaf in self.leaves:
            parts = split_ec(leaf)
            if len(parts) != 4:
                continue
            prev = "ROOT"
            for lvl in range(1, 5):
                node = node_at_level(parts, lvl)
                _children[prev].add(node)
                self.nodes_by_level[lvl].add(node)
                prev = node
        self.children = {k: sorted(v) for k, v in _children.items()}

    # -- I/O -------------------------------------------------------------- #
    @classmethod
    def from_all_ec(cls, path=ALL_EC_TXT) -> "ECTree":
        with open(path) as fh:
            leaves = [ln.strip() for ln in fh if ln.strip()]
        return cls(leaves)

    @classmethod
    def load(cls, path=EC_TREE_JSON) -> "ECTree":
        with open(path) as fh:
            blob = json.load(fh)
        tree = cls.__new__(cls)
        tree.children = {k: list(v) for k, v in blob["children"].items()}
        tree.nodes_by_level = {int(k): set(v) for k, v in blob["nodes_by_level"].items()}
        tree.leaves = list(blob["leaves"])
        return tree

    def save(self, path=EC_TREE_JSON) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "children": self.children,
            "nodes_by_level": {str(k): sorted(v) for k, v in self.nodes_by_level.items()},
            "leaves": self.leaves,
            "stats": self.stats(),
        }
        with open(path, "w") as fh:
            json.dump(blob, fh)

    def stats(self) -> dict:
        return {
            "n_leaves": len(self.leaves),
            "n_level1": len(self.nodes_by_level.get(1, [])),
            "n_level2": len(self.nodes_by_level.get(2, [])),
            "n_level3": len(self.nodes_by_level.get(3, [])),
            "n_level4": len(self.nodes_by_level.get(4, [])),
        }

    def get_children(self, node: str) -> list[str]:
        return self.children.get(node, [])


# --------------------------------------------------------------------------- #
# Shared candidate-set builder (used by item-gen AND cascade eval)
# --------------------------------------------------------------------------- #
def build_candidates(
    tree: ECTree,
    parents: Iterable[str],
    gold_options: Iterable[str],
    level: int,
    n_max: int,
    rng: random.Random,
    hard_ratio: float = 0.7,
) -> tuple[list[str], float]:
    """Construct a (shuffled) candidate list for one cascade step.

    * `parents`     - the parent node(s) whose children form the option universe.
                      gold parent(s) when generating items; the chosen node at eval.
    * `gold_options`- correct nodes at this level (>=1 for multi-label); always shown.
    * hard distractors = siblings (other children of the parents);
      easy distractors = random nodes at this level from other subtrees.

    Returns (candidates_without_UNSURE, realized_hard_distractor_share).
    """
    gold = list(dict.fromkeys(gold_options))
    sibling = set()
    for p in parents:
        sibling.update(tree.get_children(p))
    sibling.difference_update(gold)
    all_lvl = tree.nodes_by_level.get(level, set())
    easy_pool = list(all_lvl - sibling - set(gold))
    sibling = list(sibling)

    n_distract = max(0, n_max - len(gold))
    n_hard = min(len(sibling), int(round(hard_ratio * n_distract)))
    n_easy = min(len(easy_pool), n_distract - n_hard)
    # backfill if one pool is short
    if n_hard + n_easy < n_distract:
        if len(sibling) > n_hard:
            n_hard = min(len(sibling), n_distract - n_easy)
        elif len(easy_pool) > n_easy:
            n_easy = min(len(easy_pool), n_distract - n_hard)

    rng.shuffle(sibling)
    rng.shuffle(easy_pool)
    chosen_hard = sibling[:n_hard]
    chosen_easy = easy_pool[:n_easy]

    cands = list(dict.fromkeys(gold + chosen_hard + chosen_easy))
    rng.shuffle(cands)
    denom = max(1, len(chosen_hard) + len(chosen_easy))
    hard_share = len(chosen_hard) / denom
    return cands, hard_share


if __name__ == "__main__":
    t = ECTree.from_all_ec()
    t.save()
    print("Saved EC tree:", json.dumps(t.stats()))
