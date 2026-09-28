#!/usr/bin/env python3
"""Prepare the public benchmark artifacts from a private working tree.

This maintainer utility performs three release-only operations:

1. reconstruct the ESM-kNN cache from the upstream embedding array;
2. pseudonymize query and retrieval-neighbor identifiers consistently;
3. remove free-form model text from prediction records.

It never writes to the source working tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecreason.identifiers import public_query_id, public_reference_id

SPLITS = ("30", "30-50", "price", "promiscuous")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def read_training_labels(path: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                labels[row[0]] = [ec.strip() for ec in row[1].split(";") if ec.strip()]
    return labels


def build_esm_cache(source_root: Path, topk: int = 5, batch_size: int = 64) -> dict[str, dict]:
    data_dir = source_root / "sourcedata" / "data" / "care_graph"
    with (data_dir / "entry2index.pkl").open("rb") as handle:
        entry_to_index = {str(key): int(value) for key, value in pickle.load(handle).items()}
    index_to_entry = {value: key for key, value in entry_to_index.items()}
    training_labels = read_training_labels(data_dir / "protein_train50.csv")
    pool_indices = np.asarray(
        sorted(entry_to_index[entry] for entry in training_labels if entry in entry_to_index),
        dtype=np.int64,
    )

    features = np.load(data_dir / "feats.npy", mmap_mode="r")
    pool = np.asarray(features[pool_indices], dtype=np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8

    cache: dict[str, dict] = {}
    for split in SPLITS:
        source_items = read_jsonl(source_root / "out" / "items" / f"{split}.jsonl")
        present = [(item["entry"], entry_to_index[item["entry"]])
                   for item in source_items if item["entry"] in entry_to_index]
        split_cache: dict[str, list[dict]] = {}
        for start in range(0, len(present), batch_size):
            batch = present[start:start + batch_size]
            query_indices = [index for _, index in batch]
            query = np.asarray(features[query_indices], dtype=np.float32)
            query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
            similarities = query @ pool.T
            for row_index, (entry, _) in enumerate(batch):
                scores = similarities[row_index]
                selected = np.argpartition(-scores, topk - 1)[:topk]
                selected = selected[np.argsort(-scores[selected])]
                hits = []
                for local_index in selected:
                    original_reference = index_to_entry[int(pool_indices[local_index])]
                    hits.append({
                        "entry": public_reference_id(original_reference),
                        "ecs": training_labels.get(original_reference, []),
                        "score": round(float(scores[local_index]), 4),
                    })
                split_cache[entry] = hits
        cache[split] = split_cache
        print(f"[prepare] ESM-kNN {split}: {len(split_cache)} queries")
    return cache


def transform_items(source_root: Path, release_root: Path) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    decontamination = []
    for split in SPLITS:
        source_rows = read_jsonl(source_root / "out" / "items" / f"{split}.jsonl")
        release_rows = read_jsonl(release_root / "out" / "items" / f"{split}.jsonl")
        if len(source_rows) != len(release_rows):
            raise RuntimeError(f"Item-count mismatch for split {split}")
        mapping = {}
        for source_row, release_row in zip(source_rows, release_rows):
            original = source_row["entry"]
            public = public_query_id(split, original)
            mapping[original] = public
            release_row["entry"] = public
            release_row["id"] = f"{split}::{public}"
            decontamination.append({
                "id": release_row["id"],
                "split": split,
                "sequence_sha256": hashlib.sha256(
                    release_row["sequence"].encode("ascii")
                ).hexdigest(),
            })
        write_jsonl(release_root / "out" / "items" / f"{split}.jsonl", release_rows)
        mappings[split] = mapping
    write_jsonl(release_root / "out" / "items" / "decontamination.jsonl", decontamination)
    return mappings


def transform_evidence(
    release_root: Path,
    mappings: dict[str, dict[str, str]],
    esm_cache: dict[str, dict],
) -> None:
    evidence_dir = release_root / "out" / "evidence"
    for path in sorted(evidence_dir.glob("*.jsonl")):
        split = path.name.split(".", 1)[0]
        rows = read_jsonl(path)
        for row in rows:
            original_query = row["entry"]
            row["entry"] = mappings[split][original_query]
            for channel, hits in list(row.items()):
                if channel == "entry" or not isinstance(hits, list):
                    continue
                for hit in hits:
                    if "entry" in hit and not hit["entry"].startswith("R-"):
                        hit["entry"] = public_reference_id(hit["entry"])
            if path.name == f"{split}.jsonl":
                # The released protocol exposes ESM neighbors only for queries that
                # are represented in the upstream inductive structure graph. This
                # reproduces the 1,101-query ESM coverage reported in the paper.
                row["esm_knn"] = (
                    esm_cache[split].get(original_query, [])
                    if row.get("structure") else []
                )
        write_jsonl(path, rows)


def enforce_published_esm_coverage(release_root: Path) -> None:
    """Apply the paper's ESM availability mask to an already public release."""
    evidence_dir = release_root / "out" / "evidence"
    for split in SPLITS:
        path = evidence_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            if not row.get("structure"):
                row["esm_knn"] = []
        write_jsonl(path, rows)


def transform_predictions(release_root: Path) -> None:
    for path in sorted((release_root / "out" / "results").glob("pred_*.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            split, original = row["id"].split("::", 1)
            row["id"] = f"{split}::{public_query_id(split, original)}"
            row.pop("raw", None)
            row.pop("error", None)
        write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    if source_root == release_root:
        raise RuntimeError("Source and release roots must be different")

    esm_cache = build_esm_cache(source_root)
    mappings = transform_items(source_root, release_root)
    transform_evidence(release_root, mappings, esm_cache)
    transform_predictions(release_root)

    provenance = {
        "benchmark": "EC-Reason-Bench",
        "n_items": sum(len(mapping) for mapping in mappings.values()),
        "splits": {split: len(mappings[split]) for split in SPLITS},
        "pseudonymization": "SHA-256 namespaced identifiers, 12 hexadecimal characters",
        "prediction_free_text_removed": True,
        "esm_topk": 5,
        "raw_preconstruction_data_included": True,
        "raw_training_items": 28182,
        "raw_test_items": 1349,
        "generated_large_intermediates_included": False,
    }
    (release_root / "metadata" / "release_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print("[prepare] Public artifacts prepared")


if __name__ == "__main__":
    main()
