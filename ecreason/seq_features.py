"""Offline sequence-derived descriptors (no API, no GPU).

The raw amino-acid sequence is the *query*. A text LLM cannot really "read" a 400-residue
string, so — mirroring PFUA's "sequence basic properties" tool — we precompute cheap,
genuinely informative descriptors the model CAN reason over (length, MW, hydropathy,
charge, composition, low-complexity, membrane propensity). These are intrinsic to the
sequence (no database lookup), so they belong to the *closed-book* setting.
"""
from __future__ import annotations

import math
from collections import Counter

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# average residue masses (Da)
W = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "E": 129.12,
    "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
    "M": 131.20, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
    "Y": 163.18, "V": 99.13,
}
HYDROPHOBIC = set("AILMFWVC")
POS, NEG = set("KR"), set("DE")


def summarize(seq: str) -> dict:
    seq = (seq or "").upper()
    n = max(1, len(seq))
    comp = Counter(c for c in seq if c in KD)
    mw = sum(W.get(c, 0.0) for c in seq) + 18.02
    kd_mean = sum(KD.get(c, 0.0) for c in seq) / n
    hyd = sum(1 for c in seq if c in HYDROPHOBIC) / n
    run = mx = 0
    for c in seq:
        if KD.get(c, 0) > 0:
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    pos = sum(1 for c in seq if c in POS)
    neg = sum(1 for c in seq if c in NEG)
    fracs = [comp.get(a, 0) / n for a in AA20]
    ent = -sum(p * math.log2(p) for p in fracs if p > 0)
    top = comp.most_common(3)
    distinct = sum(1 for a in AA20 if comp.get(a, 0) > 0)
    low_complex = (top and top[0][1] / n > 0.4) or distinct < 10
    return {
        "length": len(seq),
        "mol_weight_kda": round(mw / 1000.0, 1),
        "kd_hydropathy_mean": round(kd_mean, 2),
        "hydrophobic_frac": round(hyd, 2),
        "max_hydrophobic_run": mx,
        "pos_frac": round(pos / n, 2),
        "neg_frac": round(neg / n, 2),
        "net_charge": pos - neg,
        "aa_entropy_bits": round(ent, 2),
        "low_complexity": bool(low_complex),
        "top3_aa": [f"{a}:{round(c / n * 100)}%" for a, c in top],
        "membrane_propensity": "high" if mx >= 18 else ("medium" if mx >= 12 else "low"),
    }


def to_prompt_lines(f: dict) -> list[str]:
    return [
        f"- length={f['length']} aa (~{f['mol_weight_kda']} kDa)",
        f"- hydropathy(KD) mean={f['kd_hydropathy_mean']}, hydrophobic_frac={f['hydrophobic_frac']}, "
        f"max_hydrophobic_run={f['max_hydrophobic_run']} (membrane propensity: {f['membrane_propensity']})",
        f"- charge: pos_frac={f['pos_frac']}, neg_frac={f['neg_frac']}, net={f['net_charge']}",
        f"- composition: top {', '.join(f['top3_aa'])}; AA entropy={f['aa_entropy_bits']} bits; "
        f"low_complexity={f['low_complexity']}",
    ]
