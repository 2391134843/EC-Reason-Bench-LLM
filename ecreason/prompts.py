"""Prompt construction for one cascade step (training-free T1/T2; model-agnostic).

The QUERY is always the enzyme's amino-acid sequence. Every prompt shows:
  * [Query sequence]          - the raw sequence (the long string), always.
  * [Sequence-derived features] - intrinsic descriptors (closed-book; always).
  * [Retrieval evidence]      - ESM-kNN / structure / active-site hits (open-book only).
Output contract is intentionally simple to parse:
    last line ->  ANSWER=<letter> ; CONF=<0..1>
"""
from __future__ import annotations

from . import seq_features as sf

EC_CLASS_NAMES = {
    "1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
    "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases",
}

SYSTEM = (
    "You are an expert enzyme function annotator. We locate an enzyme's EC number by "
    "walking the EC tree top-down. At each level you pick exactly one option from a "
    "closed candidate list (all candidates are valid EC nodes). Reason from the query "
    "sequence, its derived properties, and any retrieval evidence. If the information is "
    "insufficient to choose reliably at this level, pick the UNSURE option to stop."
)

MAX_SEQ_CHARS = 1100  # > p95 (800) and >= max (1023); long sequences get a truncation note


def render_sequence(sequence: str | None) -> str:
    if not sequence:
        return "(sequence unavailable)"
    if len(sequence) > MAX_SEQ_CHARS:
        return f"{sequence[:MAX_SEQ_CHARS]}...[truncated, full length={len(sequence)} aa]"
    return sequence


def render_seq_features(features: dict | None, sequence: str | None) -> str:
    f = features or (sf.summarize(sequence) if sequence else None)
    if not f:
        return "(none)"
    return "\n".join(sf.to_prompt_lines(f))


def render_retrieval(ev: dict | None, max_items: int = 5) -> str:
    if not ev:
        return "(closed-book: no retrieval evidence; rely on the sequence and its features)"
    lines = []
    for kind in ("blast", "hmmer", "esm_knn", "seq_homology", "structure", "active_site", "pfam"):
        hits = ev.get(kind) or []
        for h in hits[:max_items]:
            ecs = ",".join(h.get("ecs", []))
            sim = h.get("identity", h.get("weight", h.get("score", "")))
            extra = f"; id={h['identity']}%" if "identity" in h else ""
            lines.append(f"- {kind} -> {h.get('entry', '?')} (EC={ecs}; sim={sim}{extra})")
    return "\n".join(lines) if lines else "(no retrieval neighbors found for this enzyme)"


def build_step_prompt(level: int, parent: str, options: list[dict], evidence: dict | None = None,
                      sequence: str | None = None, seq_features: dict | None = None,
                      show_sequence: bool = True) -> str:
    parent_txt = "ROOT (top level)" if parent == "ROOT" else parent
    if level == 1:
        parent_txt = "ROOT — choose among the 7 main EC classes"
    opt_lines = []
    for o in options:
        name = ""
        if o["node"] != "UNSURE":
            head = o["node"].split(".")[0]
            if level == 1 and head in EC_CLASS_NAMES:
                name = f"  ({EC_CLASS_NAMES[head]})"
        label = o["node"] if o["node"] != "UNSURE" else "UNSURE (insufficient evidence — stop here)"
        opt_lines.append(f"  {o['label']}) {label}{name}")
    seq_block = f"\n[Query sequence]\n{render_sequence(sequence)}\n" if show_sequence else ""
    return (
        f"{SYSTEM}\n"
        f"{seq_block}"
        f"\n[Sequence-derived features]\n{render_seq_features(seq_features, sequence)}\n"
        f"\n[Retrieval evidence]\n{render_retrieval(evidence)}\n"
        f"\n[Current level] Level {level}; parent = {parent_txt}\n"
        f"[Candidates]\n" + "\n".join(opt_lines) + "\n"
        f"\n[Instructions]\n"
        f"1) In 2-4 sentences, reason biochemically (substrate/cofactor/mechanism/evidence).\n"
        f"2) Optionally score plausible candidates 1-10.\n"
        f"3) FINAL line MUST be exactly:  ANSWER=<letter> ; CONF=<0..1>\n"
    )


RL_SYSTEM = (
    "You are an expert enzyme function annotator. Given a query enzyme, predict its EC "
    "number by walking the EC tree top-down. Reason briefly, then output the path, one "
    "line per level, using VALID EC nodes with dashes for unfilled positions:\n"
    "    L1: a.-.-.-\n    L2: a.b.-.-\n    L3: a.b.c.-\n    L4: a.b.c.d\n"
    "If you cannot reliably go deeper, write 'L<k>: STOP' instead of guessing.\n"
    "EC main classes: 1=Oxidoreductases 2=Transferases 3=Hydrolases 4=Lyases "
    "5=Isomerases 6=Ligases 7=Translocases."
)


def build_rl_prompt(sequence: str | None, seq_features: dict | None = None,
                    evidence: dict | None = None, show_sequence: bool = True) -> str:
    """Single-shot full-path prompt for RL / SFT (the policy emits the whole L1..L4 walk)."""
    seq_block = f"\n[Query sequence]\n{render_sequence(sequence)}\n" if show_sequence else ""
    ev_block = ""
    if evidence:
        ev_block = f"\n[Retrieval evidence]\n{render_retrieval(evidence)}\n"
    return (
        f"{RL_SYSTEM}\n"
        f"{seq_block}"
        f"\n[Sequence-derived features]\n{render_seq_features(seq_features, sequence)}\n"
        f"{ev_block}"
        f"\n[Output] First 2-4 sentences of reasoning, then the 4 EC path lines."
    )


def gold_path_lines(ec: str) -> str:
    """Render the supervised target path for one EC, e.g. '1.8.1.9' -> 4 L-lines."""
    p = ec.split(".")
    return "\n".join(
        f"L{lvl}: " + ".".join(p[:lvl] + ["-"] * (4 - lvl)) for lvl in range(1, 5)
    )


def parse_answer(text: str, valid_labels: list[str]) -> tuple[str | None, float]:
    """Extract (label, conf) from a model response. Robust to minor format drift."""
    import re

    label, conf = None, 0.5
    m = re.search(r"ANSWER\s*=\s*([A-Za-z]{1,3})", text)
    if m:
        cand = m.group(1).upper()
        if cand in valid_labels:
            label = cand
    c = re.search(r"CONF\s*=\s*([01](?:\.\d+)?)", text)
    if c:
        try:
            conf = max(0.0, min(1.0, float(c.group(1))))
        except ValueError:
            pass
    if label is None:  # fallback: last standalone letter token that is a valid label
        for tok in reversed(re.findall(r"\b([A-Z]{1,3})\b", text.upper())):
            if tok in valid_labels:
                label = tok
                break
    return label, conf
