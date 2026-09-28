"""Centralized filesystem paths.

Resolved relative to this file so the package works regardless of the CWD.
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../benchmark-main/ecreason
BENCH_DIR = PKG_DIR.parent                          # .../benchmark-main
REPO_ROOT = BENCH_DIR.parent                        # .../benchmark (repo root)

# Original sequence tables are vendored under benchmark-main/sourcedata/data.
# Generated arrays, maps, and graphs may be placed in the same CARE directory
# when rebuilding evidence. An environment override can point to a full upstream
# data directory when needed.
DATA_DIR = Path(os.environ.get("ECREASON_DATA_DIR", BENCH_DIR / "sourcedata" / "data"))
CARE_DIR = DATA_DIR / "care_graph"

ALL_EC_TXT = DATA_DIR / "all_ec.txt"
TRAIN_CSV = CARE_DIR / "protein_train50.csv"
SPLIT_CSV = {
    "30": CARE_DIR / "30_protein_test.csv",
    "30-50": CARE_DIR / "30-50_protein_test.csv",
    "price": CARE_DIR / "price_protein_test.csv",
    "promiscuous": CARE_DIR / "promiscuous_protein_test.csv",
}

EC2IDX_PKL = CARE_DIR / "ec2idx.pkl"
ENTRY2IDX_PKL = CARE_DIR / "entry2index.pkl"
LABELS_NPY = CARE_DIR / "labels.npy"
FEATS_NPY = CARE_DIR / "feats.npy"
ADJ_FOLDSEEK_NPZ = CARE_DIR / "adj_foldseek_nbit_03_inductive.npz"
ADJ_DISCO_NPZ = CARE_DIR / "adj_disco_005_07_inductive.npz"

# Generated artifacts live under benchmark-main/out/
OUT_DIR = BENCH_DIR / "out"
EC_TREE_JSON = OUT_DIR / "ec_tree.json"
ITEMS_DIR = OUT_DIR / "items"
EVIDENCE_DIR = OUT_DIR / "evidence"
RESULTS_DIR = OUT_DIR / "results"


def ensure_out_dirs() -> None:
    for d in (OUT_DIR, ITEMS_DIR, EVIDENCE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
