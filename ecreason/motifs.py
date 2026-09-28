"""Offline catalytic / cofactor-binding sequence-motif scanner (no API, no GPU, no DB).

The mechanism-grounded chain-of-thought method (M3, see ``mechanism_cot.py``) needs to
anchor each reasoning step in something the text LLM can actually *verify* on the raw
sequence — otherwise the "reasoning" degenerates into surface-level verbalization
(exactly the failure PFUA, arXiv:2601.03604, reports for unconstrained protein CoT).

We therefore scan for a small set of well-characterized, textbook catalytic / binding
signatures. Each is a *weak prior* on the EC class (false positives DO occur, especially
in long sequences), not a determinant — the prompt is explicit about this. Crucially this
is **intrinsic** to the sequence: no database lookup, no homology search, so it stays in
the *closed-book* setting (this is what makes M3 distinct from the retrieval track M2).

References for the motifs themselves:
  * Rossmann dinucleotide-binding GxGxxG  — Rossmann et al., Nature 1974; Kleiger & Eisenberg 2002.
  * Walker-A / P-loop  [AG]xxxxGK[ST]      — Walker et al., EMBO J. 1982.
  * alpha/beta-hydrolase nucleophile elbow GxSxG — Ollis et al., Protein Eng. 1992.
  * Zinc-metallopeptidase HExxH            — Vallee & Auld, Biochemistry 1990.
  * Aspartic-protease catalytic D[TS]G     — Davies, Annu. Rev. Biophys. 1990.
  * Thioredoxin redox active site CxxC (CGPC/CGHC) — Holmgren, JBC 1989.
"""
from __future__ import annotations

import re

# (name, compiled pattern, human interpretation, EC-class hint shown to the model)
# Patterns are deliberately conservative; `ec_hint` names the class(es) the motif *suggests*.
_MOTIFS: list[tuple[str, re.Pattern, str, str]] = [
    ("Rossmann (GxGxxG)", re.compile(r"G.G..[GA]"),
     "dinucleotide (NAD(P)/FAD) binding fold",
     "oxidoreductase (EC 1); also SAM-binding methyltransferase (EC 2.1.1)"),
    ("Walker-A P-loop ([AG]x4GK[ST])", re.compile(r"[AG].{4}GK[ST]"),
     "phosphate-binding loop for ATP/GTP",
     "kinase/phosphotransferase (EC 2.7), ATPase/translocase (EC 7, EC 3.6), or ligase (EC 6)"),
    ("alpha/beta-hydrolase elbow (GxSxG)", re.compile(r"G.S.G"),
     "nucleophilic serine in an alpha/beta-hydrolase fold",
     "serine hydrolase: esterase/lipase (EC 3.1) or serine peptidase (EC 3.4.21)"),
    ("Zn-metallopeptidase (HExxH)", re.compile(r"HE..H"),
     "two-His zinc-binding catalytic motif",
     "metal-dependent hydrolase/peptidase (EC 3.4.24); occasionally lyase (EC 4)"),
    ("Aspartic-protease (D[TS]G)", re.compile(r"D[TS]G"),
     "catalytic aspartate dyad context",
     "aspartic peptidase (EC 3.4.23)"),
    ("Thioredoxin redox (C[GP][PH]C)", re.compile(r"C[GP][PH]C"),
     "vicinal-cysteine redox active site",
     "thiol-disulfide oxidoreductase (EC 1.8) / protein-disulfide isomerase (EC 5.3.4)"),
]


def scan(sequence: str | None, max_per_motif: int = 1) -> list[dict]:
    """Return detected motifs as a list of dicts (in sequence order of first hit).

    Each dict: {name, pos (1-based), count, meaning, ec_hint}. `pos` is the start of the
    first occurrence; `count` is the total number of (overlapping) occurrences found.
    """
    seq = (sequence or "").upper()
    hits: list[dict] = []
    for name, pat, meaning, ec_hint in _MOTIFS:
        # findall with overlap via finditer on a lookahead-wrapped pattern
        positions = [m.start() + 1 for m in re.finditer(f"(?={pat.pattern})", seq)]
        if positions:
            hits.append({
                "name": name,
                "pos": positions[0],
                "count": len(positions),
                "meaning": meaning,
                "ec_hint": ec_hint,
            })
    hits.sort(key=lambda h: h["pos"])
    return hits


def to_prompt_lines(hits: list[dict]) -> list[str]:
    if not hits:
        return ["- (no canonical catalytic motif detected; rely on global composition/features)"]
    lines = []
    for h in hits:
        occ = f", {h['count']}x" if h["count"] > 1 else ""
        lines.append(f"- {h['name']} @res {h['pos']}{occ} -> {h['meaning']} => hints {h['ec_hint']}")
    return lines


def summarize_for_item(sequence: str | None) -> dict:
    """Compact, JSON-serializable motif summary suitable for caching inside an item."""
    hits = scan(sequence)
    return {
        "n_motifs": len(hits),
        "names": [h["name"] for h in hits],
        "ec_hints": sorted({h["ec_hint"] for h in hits}),
        "hits": hits,
    }
