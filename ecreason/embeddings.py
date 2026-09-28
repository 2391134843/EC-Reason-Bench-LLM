"""Accessor for the precomputed ESM-2 sequence embeddings (`feats.npy`).

These 1280-d vectors *encode the sequence* (ESM-2 650M). Two uses:
  1. ESM-kNN retrieval evidence (see evidence.py) — a strong, fully-offline,
     sequence-based "nearest known enzyme" signal (no BLAST needed).
  2. Optional multimodal injection (plan §7.5): feed an enzyme's embedding to a 7B via
     a small adapter as soft tokens, so the model can actually "see" the sequence (the
     Pika idea). `get_embedding(entry)` returns the raw vector for that pathway.
"""
from __future__ import annotations

import functools

import numpy as np

from .data_utils import load_entry2idx
from .paths import FEATS_NPY


@functools.lru_cache(maxsize=1)
def _feats() -> np.ndarray:
    return np.load(FEATS_NPY, mmap_mode="r")


@functools.lru_cache(maxsize=1)
def _normed() -> np.ndarray:
    f = np.asarray(_feats(), dtype=np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def get_embedding(entry: str) -> np.ndarray | None:
    e2i = load_entry2idx()
    idx = e2i.get(entry)
    if idx is None:
        return None
    return np.asarray(_feats()[idx]).copy()


def cosine_knn(query_idx: list[int], pool_idx: list[int], topk: int) -> list[list[tuple[int, float]]]:
    """For each query row, return [(pool_global_idx, cosine), ...] top-k over the pool."""
    norm = _normed()
    pool = np.asarray(pool_idx)
    pool_mat = norm[pool]                      # (Npool, d)
    out = []
    for qi in query_idx:
        sims = pool_mat @ norm[qi]             # (Npool,)
        k = min(topk, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out.append([(int(pool[j]), float(sims[j])) for j in top])
    return out
