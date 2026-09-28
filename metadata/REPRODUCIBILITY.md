# Reproducibility Notes

## Fixed benchmark parameters

| Parameter | Value |
|---|---:|
| Valid EC leaves | 4,963 |
| Test items | 1,349 |
| Candidate cap below level 1 | 10 |
| Sibling-distractor target share | 0.70 |
| Construction seed | 0 |
| M2 global retrieval budget | 4 calls per enzyme |
| M2 per-level ReAct cap | 3 turns |
| M4 samples | 3 |
| M4 prefix-vote threshold | 0.5 |
| ECE bins | 10 equal-width bins |
| Evaluation seeds | 1 |
| Bootstrap replicates in the paper | 1,000 |

Sequences are truncated only in prompts longer than 1,100 amino-acid characters;
the full sequences remain in the benchmark records.

## Included pre-construction snapshot

The original upstream sequence tables are under `sourcedata/data/` and the
constructed benchmark is under `out/`.

| Raw input | Records or shape |
|---|---:|
| `all_ec.txt` | 4,963 EC leaves |
| `care_graph/protein_train50.csv` | 28,182 training records |
| Four `care_graph/*_protein_test.csv` files | 1,349 test records |

Large generated arrays, maps, and sparse graphs are omitted because the
evaluation-ready evidence derived from them is already included.

Run `shasum -a 256 -c metadata/MANIFEST.sha256` to check the distributed files.
The original `scripts/verify_release.py` is retained from the full supplement;
its result-completeness check requires the omitted experiment outputs.

## Nine reported conditions

1. B0 closed-book direct generation
2. M1 closed-book hierarchical cascade
3. M3 closed-book mechanistic chain of thought
4. M4 closed-book self-consistency over M1
5. B0 passive open-book direct generation
6. M2 active retrieval over the cascade
7. Passive open-book cascade
8. M3 passive open-book mechanistic chain of thought
9. M4 passive open-book self-consistency over M1

All conditions share the same items, EC tree, evidence cache, and scorer.

## Model-prefix mapping

| Result prefix | Reported model |
|---|---|
| `ds_` | deepseek-v4-pro |
| `g55_` | gpt-5.5 |
| `q235_` | qwen3-235b-a22b-thinking |
| `g31_` | gemini-3.1-pro |
| `glm52_` | glm-5.2 |

Each condition has a `big` result over 992 items and a `small` result over 357
items. `ecreason.summarize` unions the structured predictions and re-scores all
1,349 samples.

## Result files

`out/results/` is excluded from this GitHub release. Running the evaluation
code produces files with the following names:

- `pred_<prefix><condition>_<pass>.jsonl`: per-item structured predictions;
- `metrics_<prefix><condition>_<pass>.json`: original per-pass metrics and run
  metadata;
- `summary_<prefix>.json`: recomputed full-set metrics and method diagnostics;
- `summary_<prefix>.md`: human-readable rendering of the same summary.

The full supplement retains only structured prediction fields required for
scoring. Those prediction files are not included in this repository.

## Environment

Core construction and scoring require NumPy and SciPy. Rebuilding homology
evidence additionally requires DIAMOND 2.2 or newer and HMMER 3.4 or newer.
Remote inference requires an OpenAI-compatible client and an endpoint selected by
the reviewer. Local inference requires PyTorch, Transformers, and optionally
vLLM.

Provider-side model revisions and nondeterministic serving can change fresh API
runs, so newly generated outputs may differ from the reported metrics.
