"""HMMER-based evidence channels (uses the installed HMMER 3.4 suite).

Two methods:
  * phmmer (default, OFFLINE, runnable now): profile-HMM homology search of each test
    sequence against the *training* sequence DB. A distinct evidence channel from
    DIAMOND (BLAST): HMMER's per-residue probabilistic profiles often detect remote
    homology BLAST misses. Output -> out/evidence/{split}.hmmer.jsonl  (channel "hmmer").
  * hmmscan vs Pfam-A (OPTIONAL): conserved-domain (family) evidence. Requires the
    Pfam-A.hmm database (~1.6GB; not downloaded here). `pfam_setup_hint()` documents it.
    Output -> out/evidence/{split}.pfam.jsonl  (channel "pfam").

Run:  python -m ecreason.hmmer_search --topk 5 --cpu 8            # phmmer channel
      python -m ecreason.hmmer_search --topk 5 --limit 10        # quick speed probe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict

from .build_items import SPLITS
from .data_utils import read_protein_csv
from .identifiers import public_query_id, public_reference_id
from .paths import EVIDENCE_DIR, SPLIT_CSV, TRAIN_CSV, ensure_out_dirs


def find_binary(name: str) -> str:
    cand = shutil.which(name)
    if cand:
        return cand
    raise RuntimeError(f"{name} not found. conda install -c bioconda hmmer")


def _write_fasta(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(f">{r['entry']}\n{r['sequence']}\n")


def _parse_tblout(path: str):
    """phmmer --tblout: col0=target(train entry), col2=query(test entry), col4=Evalue, col5=score."""
    hits = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 6:
                continue
            target, query, evalue, score = f[0], f[2], f[4], f[5]
            hits[query].append({"entry": target, "evalue": evalue, "score": float(score)})
    return hits


def phmmer_split(phmmer, train_fasta, ec_of, split, topk, cpu, limit=0, evalue="1e-3"):
    rows = read_protein_csv(SPLIT_CSV[split])
    if limit:
        rows = rows[:limit]
    with tempfile.TemporaryDirectory() as td:
        qfa = os.path.join(td, "q.faa")
        tbl = os.path.join(td, "out.tbl")
        _write_fasta(rows, qfa)
        subprocess.run([phmmer, "--noali", "--tblout", tbl, "-E", evalue, "--cpu", str(cpu),
                        "-o", os.devnull, qfa, train_fasta], check=True)
        hits = _parse_tblout(tbl)
    n_hit = 0
    with open(EVIDENCE_DIR / f"{split}.hmmer.jsonl", "w") as fh:
        for r in rows:
            top = sorted(hits.get(r["entry"], []), key=lambda x: -x["score"])[:topk]
            for h in top:
                h["ecs"] = ec_of.get(h["entry"], [])
                h["score"] = round(h["score"], 1)
                h["entry"] = public_reference_id(h["entry"])
            n_hit += bool(top)
            fh.write(json.dumps({
                "entry": public_query_id(split, r["entry"]),
                "hmmer": top,
            }, ensure_ascii=False) + "\n")
    return {"n": len(rows), "with_hmmer": n_hit}


def pfam_setup_hint() -> str:
    return (
        "OPTIONAL Pfam/hmmscan domain channel:\n"
        "  curl -O https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz  # ~1.6GB\n"
        "  gunzip Pfam-A.hmm.gz && hmmpress Pfam-A.hmm\n"
        "  hmmscan --tblout {split}.pfam.tbl --noali --cpu 8 Pfam-A.hmm queries.faa\n"
        "Then write out/evidence/{split}.pfam.jsonl with a 'pfam' list of domain accessions.\n"
        "NOTE: hmmscan over full Pfam for ~1.3k queries is slow (hours); phmmer is the\n"
        "default offline HMMER channel here."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap queries per split (speed probe)")
    args = ap.parse_args()
    ensure_out_dirs()
    phmmer = find_binary("phmmer")
    train = read_protein_csv(TRAIN_CSV)
    ec_of = {r["entry"]: r["ecs"] for r in train}
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False) as tf:
        train_fasta = tf.name
    _write_fasta(train, train_fasta)
    try:
        stats = {}
        for split in SPLITS:
            t0 = time.time()
            stats[split] = phmmer_split(phmmer, train_fasta, ec_of, split, args.topk, args.cpu, args.limit)
            stats[split]["seconds"] = round(time.time() - t0, 1)
            print(f"[phmmer] {split}: {json.dumps(stats[split])}", flush=True)
    finally:
        os.remove(train_fasta)
    print(json.dumps({"hmmer_evidence": stats}, indent=2))


if __name__ == "__main__":
    main()
