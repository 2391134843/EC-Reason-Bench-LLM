#!/usr/bin/env python3
"""Validate counts, joins, sanitization, and final-result completeness."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_COUNTS = {"30": 432, "30-50": 560, "price": 148, "promiscuous": 209}
RAW_SPLIT_FILES = {
    "30": "30_protein_test.csv",
    "30-50": "30-50_protein_test.csv",
    "price": "price_protein_test.csv",
    "promiscuous": "promiscuous_protein_test.csv",
}
RAW_CSV_COUNTS = {
    "protein_train50.csv": 28182,
    "30_protein_test.csv": 432,
    "30-50_protein_test.csv": 560,
    "price_protein_test.csv": 148,
    "promiscuous_protein_test.csv": 209,
}
EVIDENCE_COVERAGE = {
    "30": {"esm_knn": 248, "blast": 365, "hmmer": 376, "structure": 248, "active_site": 22},
    "30-50": {"esm_knn": 526, "blast": 551, "hmmer": 556, "structure": 526, "active_site": 55},
    "price": {"esm_knn": 127, "blast": 141, "hmmer": 143, "structure": 127, "active_site": 0},
    "promiscuous": {"esm_knn": 200, "blast": 209, "hmmer": 209, "structure": 200, "active_site": 22},
}
MODEL_PREFIXES = ("ds", "g55", "q235", "g31", "glm52")
CONDITIONS = (
    "b0_closed", "b0_open", "m1", "m2", "casc_open",
    "m3", "m3_open", "m4", "m4_open",
)
TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".csv", ".json", ".jsonl", ".yaml", ".yml",
    ".example", ".gitignore",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_public_query_id(split: str, original_entry: str) -> str:
    namespace = f"query:{split}"
    payload = f"ec-reason-bench-v1\0{namespace}\0{original_entry}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12].upper()
    split_tag = split.replace("-", "_").replace("%", "")
    return f"Q-{split_tag}-{digest}"


def validate_raw_inputs() -> dict[str, dict[str, dict]]:
    data_dir = ROOT / "sourcedata" / "data"
    care_dir = data_dir / "care_graph"

    ec_leaves = {
        line.strip()
        for line in (data_dir / "all_ec.txt").read_text().splitlines()
        if line.strip()
    }
    assert len(ec_leaves) == 4963

    all_entries: set[str] = set()
    parsed_tables: dict[str, list[list[str]]] = {}
    for filename, expected in RAW_CSV_COUNTS.items():
        with (care_dir / filename).open(newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            assert next(reader, None) == ["Entry", "EC number", "Sequence"]
            rows = [row for row in reader if row]
        assert len(rows) == expected, (filename, len(rows), expected)
        assert all(len(row) >= 3 and row[0] and row[1] and row[2] for row in rows)
        entries = {row[0] for row in rows}
        assert len(entries) == expected, f"Duplicate entry in {filename}"
        assert all_entries.isdisjoint(entries), f"Cross-split entry overlap in {filename}"
        all_entries.update(entries)
        parsed_tables[filename] = rows
    assert len(all_entries) == 29531

    raw_splits: dict[str, dict[str, dict]] = {}
    for split, filename in RAW_SPLIT_FILES.items():
        raw_splits[split] = {}
        for entry, ec_field, sequence, *_ in parsed_tables[filename]:
            ecs = [ec.strip() for ec in ec_field.split(";") if ec.strip()]
            assert set(ecs).issubset(ec_leaves), (split, entry, ecs)
            public_entry = expected_public_query_id(split, entry)
            raw_splits[split][public_entry] = {"sequence": sequence, "ecs": ecs}
        assert len(raw_splits[split]) == SPLIT_COUNTS[split]
    return raw_splits


def validate_items_and_evidence(raw_splits: dict[str, dict[str, dict]]) -> None:
    for split, expected in SPLIT_COUNTS.items():
        items = read_jsonl(ROOT / "out" / "items" / f"{split}.jsonl")
        assert len(items) == expected, (split, len(items))
        entries = {item["entry"] for item in items}
        assert len(entries) == expected
        assert entries == set(raw_splits[split])
        assert all(re.fullmatch(r"Q-[A-Za-z0-9_]+-[0-9A-F]{12}", entry) for entry in entries)
        assert all(item["id"] == f"{split}::{item['entry']}" for item in items)
        assert all(len(item["chain"]) == 4 for item in items)
        for item in items:
            raw = raw_splits[split][item["entry"]]
            assert item["sequence"] == raw["sequence"], item["id"]
            assert sorted(item["true_ecs"]) == sorted(raw["ecs"]), item["id"]

        channel_rows: dict[str, list[dict]] = {}
        for suffix, channels in (
            ("", ("esm_knn", "structure", "active_site")),
            (".blast", ("blast",)),
            (".hmmer", ("hmmer",)),
        ):
            rows = read_jsonl(ROOT / "out" / "evidence" / f"{split}{suffix}.jsonl")
            assert len(rows) == expected
            assert {row["entry"] for row in rows} == entries
            for channel in channels:
                channel_rows[channel] = rows
            for row in rows:
                for channel in channels:
                    assert all(
                        re.fullmatch(r"R-[0-9A-F]{12}", hit["entry"])
                        for hit in row.get(channel, [])
                    )
        for channel, coverage in EVIDENCE_COVERAGE[split].items():
            actual = sum(bool(row.get(channel)) for row in channel_rows[channel])
            assert actual == coverage, (split, channel, actual, coverage)

    decontamination = read_jsonl(ROOT / "out" / "items" / "decontamination.jsonl")
    assert len(decontamination) == sum(SPLIT_COUNTS.values())
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sequence_sha256"])
               for row in decontamination)


def validate_tree_and_results() -> None:
    tree = json.loads((ROOT / "out" / "ec_tree.json").read_text())
    assert tree["stats"] == {
        "n_leaves": 4963,
        "n_level1": 7,
        "n_level2": 73,
        "n_level3": 245,
        "n_level4": 4963,
    }
    results = ROOT / "out" / "results"
    for prefix in MODEL_PREFIXES:
        for condition in CONDITIONS:
            for pass_name, expected_n in (("big", 992), ("small", 357)):
                metrics_path = results / f"metrics_{prefix}_{condition}_{pass_name}.json"
                predictions_path = results / f"pred_{prefix}_{condition}_{pass_name}.jsonl"
                assert metrics_path.exists(), metrics_path.name
                assert predictions_path.exists(), predictions_path.name
                metrics = json.loads(metrics_path.read_text())
                predictions = read_jsonl(predictions_path)
                assert metrics["n"] == expected_n, metrics_path.name
                assert len(predictions) == expected_n, predictions_path.name
                assert all("raw" not in row and "error" not in row for row in predictions)
                assert all(re.fullmatch(
                    r"(30|30-50|price|promiscuous)::Q-[A-Za-z0-9_]+-[0-9A-F]{12}",
                    row["id"],
                ) for row in predictions)
        assert (results / f"summary_{prefix}.json").exists()
        assert (results / f"summary_{prefix}.md").exists()


def validate_sanitization() -> None:
    forbidden_path = re.compile(r"/(?:Users|root|home)/")
    han = re.compile(r"[\u3400-\u9fff]")
    high_entropy_token = re.compile(
        r"(?:sk|ghp|xoxb|AKIA|AIza)[-_]?[A-Za-z0-9]{16,}"
    )
    explicit_secret = re.compile(
        r"OPENAI_API_KEY\s*=\s*(?!REPLACE_WITH_YOUR_KEY)(?!EMPTY\b)\S+"
    )
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".env.example"}:
            continue
        text = path.read_text(errors="ignore")
        assert not han.search(text), f"Non-English CJK text in {path.relative_to(ROOT)}"
        assert not forbidden_path.search(text), f"Private absolute path in {path.relative_to(ROOT)}"
        assert not explicit_secret.search(text), f"Possible credential in {path.relative_to(ROOT)}"
        if "out" not in path.parts and "sourcedata" not in path.parts:
            assert not high_entropy_token.search(text), (
                f"Possible high-entropy token in {path.relative_to(ROOT)}"
            )


def validate_manifest() -> None:
    manifest = ROOT / "metadata" / "MANIFEST.sha256"
    assert manifest.exists(), "metadata/MANIFEST.sha256 is missing"
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), f"Manifest entry is missing: {relative}"
        assert file_sha256(path) == digest, (
            f"Checksum mismatch: {relative}"
        )
        listed.add(relative)
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    }
    assert listed == expected, "Manifest file list does not match the release"


def main() -> None:
    raw_splits = validate_raw_inputs()
    validate_items_and_evidence(raw_splits)
    validate_tree_and_results()
    validate_sanitization()
    validate_manifest()
    print("[verify] release validation passed")


if __name__ == "__main__":
    main()
