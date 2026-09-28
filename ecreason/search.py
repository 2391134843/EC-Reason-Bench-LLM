"""M2 - Intelligent Search: active, agentic retrieval over the EC-tree cascade.

This is the *method* form of the open-book setting. The plain open-book condition
(`run_eval --channels all`) simply dumps every cached evidence channel into the prompt.
Intelligent Search instead makes the model an AGENT that decides, at each cascade level:

    do I have enough evidence to choose?  -> if yes, commit.
    if not, WHICH tool should I call next? -> issue ACTION=<tool>, read the observation,
    then decide again. Searching costs budget, so the model must search selectively.

It is a ReAct-style loop (Yao et al., arXiv:2210.03629) specialized to enzyme annotation
and grounded in the SAME offline, inductive evidence cache as the rest of the benchmark
(BLAST / HMMER / ESM-kNN / structure / active-site -- all retrieved only from the training
set, no network). The contribution we measure is therefore *search strategy*, not raw
evidence availability: besides accuracy we report the search cost (tool calls per enzyme,
which tools the model relied on). It walks the EC tree like the cascade (T1), so candidates
are always tree-valid -> valid-EC = 1.0 by construction.

Maps to the paper's M2 intelligent-search method.

Run:  python -m ecreason.search --backend mock --limit 40
      python -m ecreason.search --backend openai:gpt-4o-mini --splits 30 --workers 8   # [API]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

from .build_items import SPLITS
from .ec_tree import ECTree, UNSURE
from .models import build_backend
from .paths import ITEMS_DIR, RESULTS_DIR, ensure_out_dirs
from .prompts import EC_CLASS_NAMES, parse_answer, render_seq_features, render_sequence
from .run_eval import cascade_options, filter_channels, load_evidence, load_items, subsample
from .scoring import evaluate

# Tools the agent may call; each maps to one offline evidence channel.
SEARCH_TOOLS = ["blast", "hmmer", "esm_knn", "structure", "active_site", "pfam"]
TOOL_DESC = {
    "blast": "BLAST/DIAMOND sequence-homology hits in the training DB (neighbors + their EC + % identity)",
    "hmmer": "phmmer profile-HMM homology hits (often catches remote homologs BLAST misses)",
    "esm_knn": "ESM-2 embedding nearest neighbors (homology in representation space)",
    "structure": "Foldseek structural neighbors (fold similarity even at low sequence identity)",
    "active_site": "Folddisco active-site / catalytic-pocket motif neighbors",
    "pfam": "Pfam protein-domain matches",
}
# Lenient action -> canonical-channel synonyms (models phrase tool names loosely).
TOOL_SYNONYMS = {
    "blast": "blast", "blastp": "blast", "diamond": "blast",
    "hmmer": "hmmer", "phmmer": "hmmer", "hmm": "hmmer",
    "esm": "esm_knn", "esm_knn": "esm_knn", "esmknn": "esm_knn", "embedding": "esm_knn", "knn": "esm_knn",
    "structure": "structure", "foldseek": "structure", "fold": "structure",
    "active": "active_site", "active_site": "active_site", "activesite": "active_site",
    "folddisco": "active_site", "disco": "active_site",
    "pfam": "pfam", "domain": "pfam", "hmmscan": "pfam",
}

SEARCH_SYSTEM = (
    "You are an expert enzyme function annotator equipped with ACTIVE RETRIEVAL tools. You "
    "locate an enzyme's EC number by walking the EC tree top-down; at each level you choose one "
    "option from a closed candidate list (all candidates are valid EC nodes). Before answering a "
    "level you MAY call retrieval tools to gather biological evidence about the query enzyme. "
    "Searching has a cost: retrieve only when the current evidence is insufficient, pick the most "
    "informative tool, avoid redundant calls, and stop searching once the evidence is decisive."
)

_ACTION_RE = re.compile(r"ACTION\s*[:=]\s*([A-Za-z_]+)", re.I)


def parse_action(text: str, available: list[str]) -> str | None:
    """Return the canonical tool an ACTION line requests, if it is available."""
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    tok = m.group(1).strip().lower()
    canon = TOOL_SYNONYMS.get(tok)
    if canon and canon in available:
        return canon
    for key, c in TOOL_SYNONYMS.items():  # token like "blast_search"
        if key in tok and c in available:
            return c
    return None


def _render_gathered(gathered: dict, max_items: int = 5) -> str:
    """Render evidence retrieved so far, noting empty (uninformative) channels."""
    if not gathered:
        return "(nothing retrieved yet)"
    lines = []
    for ch, hits in gathered.items():
        if not hits:
            lines.append(f"- {ch}: no neighbors found (uninformative)")
            continue
        for h in hits[:max_items]:
            ecs = ",".join(h.get("ecs", []))
            sim = h.get("identity", h.get("weight", h.get("score", "")))
            extra = f"; id={h['identity']}%" if "identity" in h else ""
            lines.append(f"- {ch} -> {h.get('entry', '?')} (EC={ecs}; sim={sim}{extra})")
    return "\n".join(lines)


def build_search_prompt(level, parent, options, gathered, available, sequence, seq_features,
                        show_sequence, budget_left, force_commit) -> str:
    parent_txt = ("ROOT — choose among the 7 main EC classes" if level == 1
                  else ("ROOT (top level)" if parent == "ROOT" else parent))
    opt_lines = []
    for o in options:
        name = ""
        if o["node"] != "UNSURE" and level == 1:
            head = o["node"].split(".")[0]
            if head in EC_CLASS_NAMES:
                name = f"  ({EC_CLASS_NAMES[head]})"
        label = o["node"] if o["node"] != "UNSURE" else "UNSURE (insufficient evidence — stop here)"
        opt_lines.append(f"  {o['label']}) {label}{name}")
    seq_block = f"\n[Query sequence]\n{render_sequence(sequence)}\n" if show_sequence else ""

    if force_commit or not available:
        tool_block = "\n[Retrieval] Search budget exhausted — you MUST commit now (no more tool calls).\n"
        decide = (
            "[Decide] Reason 2-3 sentences from the evidence already gathered, then a FINAL line:\n"
            "    ANSWER=<letter> ; CONF=<0..1>\n"
        )
    else:
        menu = "\n".join(f"  - {t}: {TOOL_DESC[t]}" for t in available)
        tool_block = (f"\n[Retrieval tools available] (search budget left: {budget_left})\n{menu}\n")
        decide = (
            "[Decide] Choose ONE:\n"
            "  (a) If the current evidence is INSUFFICIENT, retrieve one tool — output a single line:\n"
            "        ACTION=<tool_name>\n"
            "  (b) Otherwise COMMIT — reason 2-3 sentences, then a FINAL line:\n"
            "        ANSWER=<letter> ; CONF=<0..1>\n"
        )
    return (
        f"{SEARCH_SYSTEM}\n"
        f"{seq_block}"
        f"\n[Sequence-derived features]\n{render_seq_features(seq_features, sequence)}\n"
        f"{tool_block}"
        f"\n[Evidence gathered so far]\n{_render_gathered(gathered)}\n"
        f"\n[Current level] Level {level}; parent = {parent_txt}\n"
        f"[Candidates]\n" + "\n".join(opt_lines) + "\n\n"
        f"{decide}"
    )


def run_search(backend, splits, channels=None, show_seq=True, max_search=4,
               max_turns_per_level=3, n_max=10, limit=0, seed=0, workers=1):
    """Agentic retrieval cascade. `max_search` = per-enzyme tool-call budget."""
    channels = channels or ["all"]  # the *toolbox* is full; the model decides what to call
    tree = ECTree.load() if (ITEMS_DIR.parent / "ec_tree.json").exists() else ECTree.from_all_ec()
    is_mock = getattr(backend, "name", "") == "mock"

    jobs = []
    for split in splits:
        items = load_items(split)
        if limit:
            items = subsample(items, limit, split)
        ev_map = load_evidence(split)
        for it in items:
            ev_all = filter_channels(ev_map.get(it["entry"]), channels) or {}
            jobs.append((it, ev_all))

    def _predict(job):
        it, ev_all = job
        available0 = [t for t in SEARCH_TOOLS if ev_all.get(t)]
        gathered: dict[str, list] = {}
        n_search = n_turns = 0
        pred_nodes, stopped, trace = {}, None, []
        committed_conf = None
        parent = "ROOT"
        for level in range(1, 5):
            options, valid_labels, label2node, gold_labels = cascade_options(
                it, tree, parent, level, "cascade", n_max, seed)
            committed = None
            last_raw = ""
            for _t in range(max_turns_per_level):
                remaining = [t for t in available0 if t not in gathered]
                force = n_search >= max_search or not remaining
                prompt = build_search_prompt(level, parent, options, gathered, remaining,
                                             it["sequence"], it.get("seq_features"), show_seq,
                                             budget_left=max_search - n_search, force_commit=force)
                if is_mock:
                    # Oracle simulation: do a one-time selective search at L1 (so the
                    # search-cost plumbing is exercised), then commit via the gold-aware act().
                    if level == 1 and not gathered and remaining:
                        for t in remaining[: min(2, len(remaining))]:
                            gathered[t] = ev_all.get(t) or []
                            n_search += 1
                        n_turns += 1
                    backend.set_gold(sorted(gold_labels), level)
                    label, conf, last_raw = backend.act(prompt, valid_labels)
                    n_turns += 1
                    committed = (label, conf)
                    break
                last_raw = backend.generate(prompt)
                n_turns += 1
                action = None if force else parse_action(last_raw, remaining)
                label, conf = parse_answer(last_raw, valid_labels)
                wants_answer = ("ANSWER" in last_raw.upper()) and label is not None
                if action and not wants_answer:
                    gathered[action] = ev_all.get(action) or []
                    n_search += 1
                    continue
                committed = (label, conf)
                break
            if committed is None:
                committed = parse_answer(last_raw, valid_labels)
            label, conf = committed
            node = label2node.get(label, UNSURE)
            trace.append({"level": level, "parent": parent, "chosen": node, "conf": conf})
            if node == UNSURE or label is None:
                stopped = level
                break
            pred_nodes[level] = node
            committed_conf = conf
            parent = node
        return {"id": it["id"], "pred_nodes": pred_nodes, "stopped_level": stopped,
                "valid": True, "confidence": committed_conf, "n_search": n_search,
                "n_turns": n_turns, "tools": list(gathered), "trace": trace}

    def _safe(job):
        try:
            return _predict(job)
        except Exception as e:  # a 429/timeout on one enzyme must not kill the whole pass
            return {"id": job[0]["id"], "pred_nodes": {}, "stopped_level": 1, "valid": True,
                    "confidence": None, "n_search": 0, "n_turns": 0, "tools": [], "trace": [],
                    "error": str(e)[:200]}

    all_items = [it for it, _ in jobs]
    all_preds = {}
    parallel = workers > 1 and getattr(backend, "name", "").startswith("openai")
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        from ._progress import tracker
        tick = tracker(len(jobs), "intelligent-search")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(_safe, jobs):
                all_preds[p["id"]] = p
                tick()
    else:
        for job in jobs:
            p = _safe(job)
            all_preds[p["id"]] = p

    metrics = evaluate(all_items, all_preds)
    n = max(1, len(all_preds))
    tool_use = Counter(t for p in all_preds.values() for t in p["tools"])
    metrics["search_cost"] = {
        "avg_searches_per_enzyme": round(sum(p["n_search"] for p in all_preds.values()) / n, 3),
        "avg_model_turns_per_enzyme": round(sum(p["n_turns"] for p in all_preds.values()) / n, 3),
        "tool_usage": {t: tool_use.get(t, 0) for t in SEARCH_TOOLS},
        "max_search_budget": max_search,
    }
    return metrics, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", help="mock | openai:<model> | hf:<model_id>")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--max-search", type=int, default=4, help="per-enzyme tool-call budget")
    ap.add_argument("--max-turns", type=int, default=3, help="max model turns per cascade level")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ensure_out_dirs()
    backend = build_backend(args.backend)
    t0 = time.time()
    metrics, preds = run_search(backend, args.splits, show_seq=not args.no_seq,
                                max_search=args.max_search, max_turns_per_level=args.max_turns,
                                n_max=args.n_max, limit=args.limit, seed=args.seed, workers=args.workers)
    metrics["meta"] = {"method": "intelligent-search", "backend": backend.name,
                       "max_search": args.max_search, "show_sequence": not args.no_seq,
                       "splits": args.splits, "limit": args.limit, "seconds": round(time.time() - t0, 2)}
    tag = args.tag or f"search_{backend.name.replace(':', '_')}"
    (RESULTS_DIR / f"pred_{tag}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds.values()))
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in
                      ("n", "accuracy", "hierarchical", "search_cost", "meta")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
