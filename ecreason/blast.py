"""Real sequence-homology evidence via DIAMOND BLASTp (offline, inductive).

Mirrors what CARE/PoinnCARE do for the BLASTp baseline: build a protein DB from the
*training* sequences, query each test sequence, take the top hits, and report their
EC numbers + identity/e-value. This is the canonical sequence-based evidence channel.

Requires the `diamond` binary (installed via bioconda in this repo). Outputs
`out/evidence/{split}.blast.jsonl` with the same per-entry schema as evidence.py.

Run:  python -m ecreason.blast --topk 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict

from .build_items import SPLITS
from .data_utils import read_protein_csv
from .identifiers import public_query_id, public_reference_id
from .paths import EVIDENCE_DIR, OUT_DIR, SPLIT_CSV, TRAIN_CSV, ensure_out_dirs


def find_diamond() -> str:
    cand = shutil.which("diamond")
    if cand:
        return cand
    raise RuntimeError("diamond not found. conda install -c bioconda diamond")


def _write_fasta(rows: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(f">{r['entry']}\n{r['sequence']}\n")


def build_db(diamond: str, db_path: str) -> dict:
    """Build a DIAMOND DB from training sequences; map train entry -> ECs."""
    train = read_protein_csv(TRAIN_CSV)
    ec_of = {r["entry"]: r["ecs"] for r in train}
    fasta = db_path + ".faa"
    _write_fasta(train, fasta)
    subprocess.run([diamond, "makedb", "--in", fasta, "-d", db_path, "--quiet"], check=True)
    os.remove(fasta)
    return ec_of


def blast_split(diamond: str, db_path: str, ec_of: dict, split: str, topk: int,
                max_target: int = 25, threads: int = 8) -> dict:
    rows = read_protein_csv(SPLIT_CSV[split])
    with tempfile.TemporaryDirectory() as td:
        qfa = os.path.join(td, "q.faa")
        out_tsv = os.path.join(td, "out.tsv")
        _write_fasta(rows, qfa)
        # outfmt 6: qseqid sseqid pident length evalue bitscore
        subprocess.run([
            diamond, "blastp", "-q", qfa, "-d", db_path, "-o", out_tsv,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "evalue", "bitscore",
            "--max-target-seqs", str(max_target), "--quiet", "--threads", str(threads),
            "--very-sensitive",
        ], check=True)
        hits = defaultdict(list)
        with open(out_tsv) as fh:
            for line in fh:
                q, s, pident, length, evalue, bits = line.rstrip("\n").split("\t")
                hits[q].append({
                    "entry": public_reference_id(s), "ecs": ec_of.get(s, []),
                    "identity": round(float(pident), 1), "evalue": evalue,
                    "bitscore": round(float(bits), 1),
                })

    n_hit = 0
    with open(EVIDENCE_DIR / f"{split}.blast.jsonl", "w") as fh:
        for r in rows:
            top = hits.get(r["entry"], [])[:topk]   # diamond already sorts by bitscore
            n_hit += bool(top)
            fh.write(json.dumps({
                "entry": public_query_id(split, r["entry"]),
                "blast": top,
            }, ensure_ascii=False) + "\n")
    return {"n": len(rows), "with_blast": n_hit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    ensure_out_dirs()
    diamond = find_diamond()
    db_dir = OUT_DIR / "blastdb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "train50")
    print(f"[diamond] building DB at {db_path} ...")
    ec_of = build_db(diamond, db_path)
    stats = {}
    for split in SPLITS:
        print(f"[diamond] blastp on split={split} ...")
        stats[split] = blast_split(diamond, db_path, ec_of, split, args.topk, threads=args.threads)
    print(json.dumps({"blast_evidence": stats}, indent=2))


if __name__ == "__main__":
    main()
