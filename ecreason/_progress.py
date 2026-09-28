"""Lightweight, opt-in progress logging for long API runs.

Enabled only when env ``ECREASON_PROGRESS`` is set (so normal/mock runs stay quiet).
Prints `[progress] <label>: done/total (elapsed, ~ETA)` to stderr every `every` items,
which the background orchestration tees into per-condition logs so you can watch progress.
"""
from __future__ import annotations

import os
import sys
import time


def tracker(total: int, label: str, every: int = 20):
    enabled = bool(os.environ.get("ECREASON_PROGRESS"))
    state = {"done": 0, "t0": time.time()}

    def tick():
        state["done"] += 1
        d = state["done"]
        if not enabled:
            return
        if d % every == 0 or d == total:
            el = time.time() - state["t0"]
            rate = d / el if el > 0 else 0.0
            eta = (total - d) / rate if rate > 0 else 0.0
            print(f"[progress] {label}: {d}/{total}  ({el:.0f}s elapsed, ~{eta:.0f}s left, "
                  f"{rate*60:.1f}/min)", file=sys.stderr, flush=True)

    return tick
