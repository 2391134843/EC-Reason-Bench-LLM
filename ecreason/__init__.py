"""EC-Reason-Bench: hierarchy-guided LLM benchmark for enzyme (EC number) classification.

This package builds the benchmark items fully offline from the CARE/PoinnCARE
sequence tables under `sourcedata/data/`. Evaluation uses the constructed
evidence cache shipped under `out/evidence/`. No API or GPU is needed to
construct items, prepare prompts, or score existing predictions.

See `benchmark-main/README.md` for the API/GPU dependency matrix.
"""

__version__ = "0.1.0"
