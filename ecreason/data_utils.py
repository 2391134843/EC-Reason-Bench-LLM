"""Loaders for the CARE CSVs, label matrix, and entry/EC index maps."""
from __future__ import annotations

import csv
import functools
import math
import pickle
from collections import Counter

from .paths import (
    EC2IDX_PKL,
    ENTRY2IDX_PKL,
    SPLIT_CSV,
    TRAIN_CSV,
)


def read_protein_csv(path) -> list[dict]:
    """Each row -> {entry, ecs:[...], sequence}. CSV is tab-separated; ECs split on ';'."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        for r in reader:
            if not r or len(r) < 3:
                continue
            entry, ec_field, seq = r[0], r[1], r[2]
            ecs = [e.strip() for e in ec_field.split(";") if e.strip()]
            rows.append({"entry": entry, "ecs": ecs, "sequence": seq})
    return rows


def load_split(split: str) -> list[dict]:
    return read_protein_csv(SPLIT_CSV[split])


@functools.lru_cache(maxsize=1)
def train_ec_counter() -> Counter:
    """Frequency of each *full* EC number in the (50%-clustered) training set."""
    c: Counter = Counter()
    for row in read_protein_csv(TRAIN_CSV):
        for ec in row["ecs"]:
            c[ec] += 1
    return c


@functools.lru_cache(maxsize=1)
def train_entry_set() -> frozenset:
    return frozenset(row["entry"] for row in read_protein_csv(TRAIN_CSV))


def tail_score(ecs: list[str]) -> float:
    """0 (head/common) .. 1 (tail/rare). Uses the rarest of an enzyme's true ECs."""
    counter = train_ec_counter()
    max_log = math.log(max(counter.values()) + 1.0) if counter else 1.0
    freqs = [counter.get(ec, 0) for ec in ecs] or [0]
    rarest = min(freqs)
    return 1.0 - math.log(rarest + 1.0) / max_log if max_log > 0 else 1.0


@functools.lru_cache(maxsize=1)
def load_entry2idx() -> dict:
    with open(ENTRY2IDX_PKL, "rb") as fh:
        return {str(k): int(v) for k, v in pickle.load(fh).items()}


@functools.lru_cache(maxsize=1)
def load_idx2entry() -> dict:
    return {v: k for k, v in load_entry2idx().items()}


@functools.lru_cache(maxsize=1)
def load_ec2idx() -> dict:
    with open(EC2IDX_PKL, "rb") as fh:
        return {str(k): int(v) for k, v in pickle.load(fh).items()}


@functools.lru_cache(maxsize=1)
def load_idx2ec() -> dict:
    return {v: k for k, v in load_ec2idx().items()}
