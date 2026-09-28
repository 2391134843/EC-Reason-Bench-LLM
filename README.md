# EC-Reason-Bench

This repository provides code and data for the paper
**Knowledge before Reasoning: EC-Reason-Bench, a Training-Free Diagnostic
Benchmark for LLM Enzyme Classification**.

EC-Reason-Bench is a training-free diagnostic benchmark for enzyme EC-number
classification with general-purpose language models.

The package contains the benchmark implementation, the original CARE/PoinnCARE
sequence tables used before construction, all 1,349 constructed test items, the
offline retrieval cache. Experimental outputs under `out/results/` are not
included in this GitHub release. Plotting-only code, figures, generated large tensors and graphs, private
endpoint configuration, API credentials, logs, obsolete smoke runs, and
unrelated post-training experiments are intentionally excluded.

## Package contents

```text
.
├── ecreason/              Core construction, inference, retrieval, and scoring code
├── out/
│   ├── ec_tree.json       EC tree with 4,963 valid leaves
│   ├── items/             Four benchmark splits and dataset metadata
│   └── evidence/          Precomputed offline evidence for five channels
├── run/
│   ├── api/               OpenAI-compatible API launchers
│   └── local/             Hugging Face and vLLM launchers
├── scripts/               Release preparation, validation, and manifest tools
├── sourcedata/
│   ├── README.md          Raw-data inventory and identifier policy
│   └── data/
│       ├── all_ec.txt     EC nomenclature snapshot used to build the tree
│       └── care_graph/    Original training and four test sequence tables
└── metadata/              Data card, reproducibility notes, and SHA-256 manifest
```

The two data layers are intentionally separate:

- `sourcedata/data/` contains the unmodified, pre-construction sequence tables
  and EC snapshot. It retains upstream accession identifiers so the benchmark
  items can be rebuilt.
- `out/items/` and `out/evidence/` are the constructed release
  artifacts. Their query and retrieval identifiers are pseudonymized.

## Quick verification

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x run/api/*.sh run/local/*.sh
shasum -a 256 -c metadata/MANIFEST.sha256
```

Run a small offline smoke test with the deterministic mock backend:

```bash
python -m ecreason.run_eval \
  --backend mock \
  --splits 30 \
  --limit 8 \
  --channels none \
  --tag smoke
```

The mock backend verifies the pipeline only; it is not a scientific baseline.

## Summarize your evaluation results

The final evaluation was run in two complete passes per condition: the `30` and
`30-50` splits (`big`, 992 items), followed by `price` and `promiscuous`
(`small`, 357 items). After running inference to generate predictions under
`out/results/`, summarize each model with:

```bash
python -m ecreason.summarize --prefix ds_    --out summary_ds    --title deepseek-v4-pro
python -m ecreason.summarize --prefix g55_   --out summary_g55   --title gpt-5.5
python -m ecreason.summarize --prefix q235_  --out summary_q235  --title qwen3-235b-a22b-thinking
python -m ecreason.summarize --prefix g31_   --out summary_g31   --title gemini-3.1-pro
python -m ecreason.summarize --prefix glm52_ --out summary_glm52 --title glm-5.2
```

These commands require locally generated predictions; saved experimental results
are not distributed in this repository.

## Run a model

For an OpenAI-compatible endpoint:

```bash
cp run/api/.env.example run/api/.env
# Edit run/api/.env locally. Never commit it.
LIMIT=30 run/api/run_compare.sh
```

For a local Hugging Face model:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
LIMIT=30 \
CHANNELS=all \
run/local/run_eval_hf.sh
```

Backend credentials are read only from environment variables. No credential or
private endpoint is present in this release.

## Rebuild from the included source data

The original training table and four test splits are included under
`sourcedata/data/care_graph/`. Rebuild the model-ready benchmark items with:

```bash
python -m ecreason.build_items --n-max 10 --hard-ratio 0.7 --seed 0
```

The evaluation-ready evidence cache is already included under `out/evidence/`.
Large generated intermediates such as ESM features, label matrices, index maps,
and Foldseek/Folddisco graphs are intentionally omitted. Rebuilding those
channels requires regenerating the intermediates with the corresponding
upstream models and tools. DIAMOND and HMMER can rebuild their channels directly
from the included training sequences.

Construction converts raw source accessions to deterministic public identifiers
in generated items and evidence. The source tables necessarily retain upstream
accessions; prompts and published predictions do not expose them as query
identifiers.

## Scope

The paper studies B0 and methods M1--M4 under closed- and open-book settings.
The corresponding implementations are:

- B0: `ecreason.baselines`
- M1 hierarchical cascade: `ecreason.run_eval`
- M2 active retrieval: `ecreason.search`
- M3 mechanistic chain of thought: `ecreason.mechanism_cot`
- M4 prefix-vote self-consistency: `ecreason.self_consistency`
- Shared scoring and reporting: `ecreason.scoring` and `ecreason.summarize`

See [`metadata/REPRODUCIBILITY.md`](metadata/REPRODUCIBILITY.md) for the exact
conditions and parameter mapping.

## Acknowledgements

We thank the authors and maintainers of the following datasets, research methods,
and software projects for making their work available to the community.

**Datasets and benchmarks**

- **[CARE](https://github.com/jsunn-y/CARE/)** — *CARE: a Benchmark Suite for the
  Classification and Retrieval of Enzymes* (Yang et al., NeurIPS 2024). Our
  benchmark uses the upstream enzyme sequence tables, EC vocabulary, and
  evaluation splits.
- **[PoinnCARE](https://openreview.net/forum?id=dGxAYNK6JU)** — *PoinnCARE:
  Hyperbolic Multi-Modal Learning for Enzyme Classification* (Xie et al., ICLR
  2026). We acknowledge its source-data organization, evaluation setting, and
  published specialist-model reference results.

**Protein representations and retrieval tools**

- **ESM-2** — *Evolutionary-scale prediction of atomic-level protein structure
  with a language model* (Lin et al., Science 2023), for the protein
  representations used in embedding-based neighbor retrieval.
- **BLAST and DIAMOND** — for sequence-homology search; the homology-cache
  builder in this repository uses DIAMOND BLASTp against the training sequences.
- **HMMER** — for the `phmmer` sequence-homology retrieval channel.
- **Foldseek** — *Fast and accurate protein structure search with Foldseek*
  (van Kempen et al., Nature Biotechnology 2024), for structural-neighbor
  evidence.
- **Folddisco** — *Structural motif search across the protein universe with
  Folddisco* (Kim et al., Nature Biotechnology 2026), for structural-motif and
  active-site evidence.

**Inference-time methods**

- **Least-to-most prompting** — *Least-to-Most Prompting Enables Complex
  Reasoning in Large Language Models* (Zhou et al., 2022), for the decomposition
  perspective behind hierarchical prediction.
- **ReAct** — *ReAct: Synergizing Reasoning and Acting in Language Models*
  (Yao et al., 2022), for interleaving model decisions with retrieval actions.
- **Self-consistency** — *Self-Consistency Improves Chain of Thought Reasoning
  in Language Models* (Wang et al., 2022), for aggregating sampled reasoning
  paths.

**Software and infrastructure**

- **NumPy and SciPy** — for numerical computation, embedding retrieval, and
  sparse-graph processing.
- **PyTorch, Hugging Face Transformers, and vLLM** — for the optional local
  model-inference backends.

We also thank the researchers who released the enzyme-prediction baselines
discussed in the paper. Please cite the corresponding upstream work when reusing
its datasets, tools, or published results.

## Data-use terms

No new license is asserted over third-party sequence records or metadata.
Consult the upstream CARE, PoinnCARE, and source-database terms before
redistributing their data.

## 🤝 Contact

If you have any questions, please feel free to open an issue or contact us at
[linyuli@stu.pku.edu.cn](mailto:linyuli@stu.pku.edu.cn).
