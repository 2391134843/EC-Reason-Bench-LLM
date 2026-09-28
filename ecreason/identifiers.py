"""Stable pseudonymous identifiers for public benchmark artifacts.

Query and reference database identifiers are deliberately separated so a public
artifact cannot reveal a UniProt or RefSeq accession through an identifier alone.
The transformation is deterministic, which keeps independently rebuilt items,
evidence files, and predictions joinable without publishing a lookup table.
"""
from __future__ import annotations

import hashlib


def _digest(namespace: str, value: str, length: int = 12) -> str:
    payload = f"ec-reason-bench-v1\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length].upper()


def public_query_id(split: str, original_entry: str) -> str:
    """Return a stable public identifier for a benchmark query."""
    split_tag = split.replace("-", "_").replace("%", "")
    return f"Q-{split_tag}-{_digest(f'query:{split}', original_entry)}"


def public_reference_id(original_entry: str) -> str:
    """Return a stable public identifier for a retrieval-database entry."""
    return f"R-{_digest('reference', original_entry)}"
