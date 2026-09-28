"""M3 - Mechanism-Grounded Protein Chain-of-Thought for enzyme classification.

A protein-specific reasoning protocol that is deliberately distinct from the other two
training-free tracks:

  * NOT the cascade (M1): there is NO per-level multiple-choice walk over the EC tree.
    The model emits the full EC in ONE pass as free text (so, like the zero-shot baseline,
    it *can* hallucinate a non-existent EC -- we report valid-EC rate to show whether the
    biochemical scaffold alone curbs that).
  * NOT retrieval / tools (M2): closed-book by default. No BLAST / HMMER / structure / KG.

Instead M3 fixes the *reasoning structure*. The EC code a.b.c.d is used as a
mechanistic decomposition (digit1=reaction class, digit2=bond/group or donor, digit3=
acceptor/cofactor, digit4=substrate). We force the model to reason along exactly those four
biochemical axes, in order, grounding every step in INTRINSIC sequence signals: the derived
features (``seq_features.py``) and detected catalytic motifs (``motifs.py``). This directly
tests the hypothesis left open by PFUA (arXiv:2601.03604) and Bio-KCoT (arXiv:2511.08024):
can a *biochemically-structured* (rather than unconstrained, or KG/tool-grounded) chain of
thought recover EC signal with no external knowledge?

It is scored with the SAME hierarchical metrics as the cascade and the zero-shot baseline,
so all four tracks (B0 / M1 / M2 / M3) sit in one comparable table.

Run:  python -m ecreason.mechanism_cot --backend mock --limit 40
      python -m ecreason.mechanism_cot --backend openai:gpt-4o-mini --splits 30   # [API]
"""
from __future__ import annotations

import argparse
import json
import time

from . import motifs as mot
from .baselines import parse_ec
from .build_items import SPLITS
from .ec_tree import ECTree, node_at_level
from .models import build_backend
from .paths import ITEMS_DIR, RESULTS_DIR, ensure_out_dirs
from .prompts import render_retrieval, render_seq_features, render_sequence
from .run_eval import filter_channels, load_evidence, load_items, subsample
from .scoring import evaluate

MECH_SYSTEM = (
    "You are an expert enzyme mechanist assigning an Enzyme Commission (EC) number. By IUBMB "
    "definition the four digits a.b.c.d are a fixed mechanistic decomposition of catalysis:\n"
    "  digit 1 (class)        = TYPE of reaction catalysed "
    "(1 oxidoreductase, 2 transferase, 3 hydrolase, 4 lyase, 5 isomerase, 6 ligase, 7 translocase);\n"
    "  digit 2 (subclass)     = the chemical bond/group acted on, or the donor group;\n"
    "  digit 3 (sub-subclass) = the acceptor, or the cofactor/coenzyme used;\n"
    "  digit 4 (serial)       = the specific substrate.\n"
    "Reason through these four mechanistic axes IN ORDER. Ground every step in the query "
    "sequence, its derived features, and the detected catalytic motifs (treat motifs as weak "
    "priors, not proof). Do NOT try to recall a memorised sequence->EC mapping; infer the "
    "mechanism. Commit only as deep as the evidence supports."
)


def build_mechanism_prompt(sequence, seq_features=None, motif_hits=None,
                           evidence=None, show_sequence=True) -> str:
    """Single-pass mechanism-grounded CoT prompt (free-form EC output)."""
    if motif_hits is None:
        motif_hits = mot.scan(sequence)
    seq_block = f"\n[Query sequence]\n{render_sequence(sequence)}\n" if show_sequence else ""
    ev_block = f"\n[Retrieval evidence]\n{render_retrieval(evidence)}\n" if evidence else ""
    motif_block = "\n".join(mot.to_prompt_lines(motif_hits))
    return (
        f"{MECH_SYSTEM}\n"
        f"{seq_block}"
        f"\n[Sequence-derived features]\n{render_seq_features(seq_features, sequence)}\n"
        f"\n[Detected catalytic motifs (intrinsic; weak priors, no database used)]\n{motif_block}\n"
        f"{ev_block}"
        f"\n[Mechanistic reasoning — fill each slot in order, one short sentence each]\n"
        f"S1 reaction class (-> digit 1): which of the 7 classes, and the signal for it?\n"
        f"S2 bond/group or donor (-> digit 2): what bond is made/broken or group transferred?\n"
        f"S3 acceptor/cofactor (-> digit 3): NAD(P)/FAD/metal/O2/CoA/none?\n"
        f"S4 specific substrate (-> digit 4): the most likely substrate.\n"
        f"\n[Answer] After the four slots, the FINAL line MUST be exactly:\n"
        f"EC=a.b.c.d        (use ';' to separate several if the enzyme is promiscuous)"
    )


def run_mechanism_cot(backend, splits, channels=None, show_seq=True, limit=0, seed=0, workers=1):
    channels = channels or ["none"]
    tree = ECTree.load() if (ITEMS_DIR.parent / "ec_tree.json").exists() else ECTree.from_all_ec()
    leaves = set(tree.leaves)

    jobs = []
    for split in splits:
        items = load_items(split)
        if limit:
            items = subsample(items, limit, split)
        ev_map = load_evidence(split) if channels != ["none"] else {}
        for it in items:
            ev = filter_channels(ev_map.get(it["entry"]), channels) if ev_map else None
            jobs.append((it, ev))

    def _predict(job):
        it, ev = job
        motif_hits = mot.scan(it["sequence"])
        prompt = build_mechanism_prompt(it["sequence"], it.get("seq_features"),
                                        motif_hits, ev, show_seq)
        if hasattr(backend, "set_zeroshot_gold"):
            backend.set_zeroshot_gold(it["true_ecs"])
        try:
            raw = backend.generate(prompt)
        except Exception as e:  # one bad item must not kill the pass
            return {"id": it["id"], "pred_nodes": {}, "pred_ecs": [], "n_motifs": len(motif_hits),
                    "valid": False, "raw": "", "error": str(e)[:200]}
        ecs = parse_ec(raw)
        primary = ecs[0] if ecs else None
        pred_nodes = {}
        if primary:
            parts = primary.split(".")
            pred_nodes = {lvl: node_at_level(parts, lvl) for lvl in range(1, 5)}
        return {"id": it["id"], "pred_nodes": pred_nodes, "pred_ecs": ecs,
                "n_motifs": len(motif_hits), "valid": bool(primary in leaves),
                "raw": (raw or "")[:200]}

    all_items = [it for it, _ in jobs]
    all_preds = {}
    parallel = workers > 1 and getattr(backend, "name", "").startswith("openai")
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        from ._progress import tracker
        tick = tracker(len(jobs), "mechanism-cot")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(_predict, jobs):
                all_preds[p["id"]] = p
                tick()
    else:
        for job in jobs:
            p = _predict(job)
            all_preds[p["id"]] = p
    metrics = evaluate(all_items, all_preds)
    metrics["motif_coverage"] = round(
        sum(1 for p in all_preds.values() if p["n_motifs"] > 0) / max(1, len(all_preds)), 4)
    return metrics, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--channels", nargs="*", default=["none"],
                    help="none (closed-book, default & headline) | all | subset")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    t0 = time.time()
    metrics, preds = run_mechanism_cot(backend, args.splits, args.channels, show_seq=not args.no_seq,
                                       limit=args.limit, seed=args.seed, workers=args.workers)
    metrics["meta"] = {"method": "mechanism-cot", "backend": backend.name,
                       "channels": args.channels, "show_sequence": not args.no_seq,
                       "splits": args.splits, "limit": args.limit, "seconds": round(time.time() - t0, 2)}
    tag = args.tag or f"mechcot_{backend.name.replace(':', '_')}_{'+'.join(args.channels)}"
    (RESULTS_DIR / f"pred_{tag}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds.values()))
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in
                      ("n", "accuracy", "valid_ec_rate", "motif_coverage", "hierarchical", "meta")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
