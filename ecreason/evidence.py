"""Build retrieval *evidence* for the open-book / tool-augmented track (T2) — OFFLINE.

Channels (all derived from the *query sequence*, all offline, NO API / NO BLAST binary):
  * esm_knn      - nearest training enzymes by ESM-2 embedding cosine (feats.npy).
                   A strong sequence-based homology signal; the main offline substitute
                   for BLAST. (ESM-2 encodes the sequence.)
  * structure    - Foldseek structural neighbors (adj_foldseek_*.npz).
  * active_site  - Folddisco active-site motif neighbors (adj_disco_*.npz).
  * seq_homology - optional pure-Python k-mer homology (--with-kmer).

Inductive setting: a test enzyme may only retrieve *training* neighbors.
Real BLAST/MMseqs2/Pfam can be plugged in later (binaries not installed here); see
`homology_stub`.

Run:  python -m ecreason.evidence --topk 5             # esm_knn + structure + active_site
      python -m ecreason.evidence --topk 5 --with-kmer
      python -m ecreason.evidence --topk 5 --no-esm     # skip ESM (e.g. feats.npy absent)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp

from .build_items import SPLITS
from .data_utils import (
    load_idx2ec,
    load_idx2entry,
    load_entry2idx,
    load_split,
    train_entry_set,
)
from .embeddings import cosine_knn
from .identifiers import public_query_id, public_reference_id
from .paths import ADJ_DISCO_NPZ, ADJ_FOLDSEEK_NPZ, EVIDENCE_DIR, LABELS_NPY, ensure_out_dirs


def _node_ecs(labels: np.ndarray, idx: int, idx2ec: dict) -> list[str]:
    return [idx2ec[int(j)] for j in np.nonzero(labels[idx])[0]]


def _graph_neighbors(adj: sp.csr_matrix, idx: int, train_idx: set[int], topk: int):
    row = adj.getrow(idx)
    pairs = [(int(c), float(v)) for c, v in zip(row.indices, row.data) if int(c) in train_idx]
    pairs.sort(key=lambda x: -x[1])
    return pairs[:topk]


def build_offline_evidence(topk: int = 5, with_esm: bool = True) -> dict:
    e2i = load_entry2idx()
    i2e = load_idx2entry()
    i2ec = load_idx2ec()
    labels = np.load(LABELS_NPY)
    A_struct = sp.load_npz(ADJ_FOLDSEEK_NPZ).tocsr()
    A_act = sp.load_npz(ADJ_DISCO_NPZ).tocsr()
    train_idx_set = {e2i[e] for e in train_entry_set() if e in e2i}
    train_idx_list = sorted(train_idx_set)

    ensure_out_dirs()
    stats = {}
    for split in SPLITS:
        rows = load_split(split)
        present = [(r["entry"], e2i[r["entry"]]) for r in rows if r["entry"] in e2i]

        # batch ESM-kNN for all present test enzymes in this split
        esm_map: dict[str, list] = {}
        if with_esm and present:
            q_idx = [idx for _, idx in present]
            knn = cosine_knn(q_idx, train_idx_list, topk)
            for (entry, _), hits in zip(present, knn):
                esm_map[entry] = [
                    {"entry": public_reference_id(i2e[nb]),
                     "ecs": _node_ecs(labels, nb, i2ec), "score": round(s, 4)}
                    for nb, s in hits
                ]

        n_esm = n_struct = n_act = n_any = 0
        with open(EVIDENCE_DIR / f"{split}.jsonl", "w") as fh:
            for row in rows:
                entry = row["entry"]
                rec = {
                    "entry": public_query_id(split, entry),
                    "esm_knn": [],
                    "structure": [],
                    "active_site": [],
                }
                if entry in e2i:
                    idx = e2i[entry]
                    rec["esm_knn"] = esm_map.get(entry, [])
                    for nb, w in _graph_neighbors(A_struct, idx, train_idx_set, topk):
                        rec["structure"].append(
                            {"entry": public_reference_id(i2e[nb]),
                             "ecs": _node_ecs(labels, nb, i2ec), "weight": round(w, 4)})
                    for nb, w in _graph_neighbors(A_act, idx, train_idx_set, topk):
                        rec["active_site"].append(
                            {"entry": public_reference_id(i2e[nb]),
                             "ecs": _node_ecs(labels, nb, i2ec), "weight": round(w, 4)})
                n_esm += bool(rec["esm_knn"])
                n_struct += bool(rec["structure"])
                n_act += bool(rec["active_site"])
                n_any += bool(rec["esm_knn"] or rec["structure"] or rec["active_site"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[split] = {"n": len(rows), "with_esm_knn": n_esm, "with_structure": n_struct,
                        "with_active_site": n_act, "with_any_evidence": n_any}
    return stats


def build_kmer_homology(topk: int = 5, k: int = 3) -> dict:
    from collections import defaultdict

    from .data_utils import read_protein_csv
    from .paths import TRAIN_CSV

    def kmer_vec(seq: str) -> dict:
        d = defaultdict(float)
        for i in range(len(seq) - k + 1):
            d[seq[i : i + k]] += 1.0
        norm = sum(v * v for v in d.values()) ** 0.5 or 1.0
        return {kk: v / norm for kk, v in d.items()}

    train = read_protein_csv(TRAIN_CSV)
    inverted = defaultdict(list)
    for ti, r in enumerate(train):
        for kk, val in kmer_vec(r["sequence"]).items():
            inverted[kk].append((ti, val))

    stats = {}
    for split in SPLITS:
        rows = load_split(split)
        with open(EVIDENCE_DIR / f"{split}.kmer.jsonl", "w") as fh:
            for row in rows:
                qv = kmer_vec(row["sequence"])
                scores = defaultdict(float)
                for kk, val in qv.items():
                    for ti, tval in inverted.get(kk, ()):
                        scores[ti] += val * tval
                top = sorted(scores.items(), key=lambda x: -x[1])[:topk]
                hits = [{"entry": public_reference_id(train[ti]["entry"]),
                         "ecs": train[ti]["ecs"], "score": round(s, 4)}
                        for ti, s in top]
                fh.write(json.dumps({
                    "entry": public_query_id(split, row["entry"]),
                    "seq_homology": hits,
                }, ensure_ascii=False) + "\n")
        stats[split] = {"n": len(rows)}
    return stats


def homology_stub() -> str:
    """Where to plug a real BLAST/MMseqs2 or Pfam tool (binaries not installed here)."""
    return (
        "Install diamond/blastp + build a DB from protein_train50.csv, or hmmscan + Pfam-A, "
        "then write evidence/{split}.blast.jsonl / .pfam.jsonl in the same per-entry schema."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no-esm", action="store_true", help="skip ESM-kNN channel")
    ap.add_argument("--with-kmer", action="store_true", help="also build offline k-mer homology")
    args = ap.parse_args()
    stats = build_offline_evidence(args.topk, with_esm=not args.no_esm)
    print(json.dumps({"offline_evidence": stats}, indent=2))
    if args.with_kmer:
        print(json.dumps({"kmer_homology": build_kmer_homology(args.topk)}, indent=2))


if __name__ == "__main__":
    main()
