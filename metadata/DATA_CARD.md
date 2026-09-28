# EC-Reason-Bench Data Card

## Summary

EC-Reason-Bench evaluates hierarchical EC-number prediction over four enzyme
splits inherited from the CARE/PoinnCARE evaluation setting.

| Public split | Paper label | Items | Multi-label items |
|---|---|---:|---:|
| `30` | `<30%` | 432 | 0 |
| `30-50` | `30-50%` | 560 | 0 |
| `price` | adversarial corrected-label set | 148 | 3 |
| `promiscuous` | multi-functional set | 209 | 209 |
| **Total** |  | **1,349** | **212** |

Every item includes an amino-acid sequence, intrinsic sequence descriptors, one
or more complete EC labels, four tree-constrained multiple-choice steps, and a
five-bin diagnostic difficulty annotation.

## Source and provenance

The split definitions and source records follow:

1. Yang et al., *CARE: a Benchmark Suite for the Classification and Retrieval of
   Enzymes*, NeurIPS 2024 Datasets and Benchmarks,
   <https://github.com/jsunn-y/CARE/>.
2. Xie et al., *PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme
   Classification*, ICLR 2026,
   <https://openreview.net/forum?id=dGxAYNK6JU>.

The release includes the two data layers needed for independent inspection and
item reconstruction:

- `sourcedata/data/all_ec.txt` is the 4,963-leaf EC snapshot;
- `sourcedata/data/care_graph/` contains the original four test tables and the
  28,182-row training table;
- `out/items/` and `out/evidence/` contain constructed benchmark records and
  the retrieval cache. Experimental outputs under `out/results/` are excluded.

Generated ESM features, label matrices, index maps, and sparse structure graphs
are omitted to keep the review package compact. They are not needed to reproduce
inference or scoring because the constructed evidence cache is included.

No new license is asserted over third-party sequence records or metadata.
Downstream users must review and comply with the upstream data terms.

## Item schema

Each line of `out/items/<split>.jsonl` is a JSON object with:

- `id`: public item identifier, unique across the release;
- `split`: one of the four public split names;
- `entry`: public query identifier used to join evidence;
- `sequence`: amino-acid sequence;
- `seq_features`: offline descriptors used by closed-book prompts;
- `true_ecs`: one or more complete EC labels;
- `n_labels`: label count;
- `chain`: four candidate sets with gold nodes and labels;
- `difficulty`: four normalized axes, composite score, and D1--D5 bin.

The constructed `entry` and all retrieval-neighbor identifiers are deterministic
pseudonyms. The release does not contain a pseudonym-to-accession map. The raw
pre-construction tables retain their upstream accessions for provenance and item
reconstruction.

## Evidence schema and coverage

Evidence is stored per query under `out/evidence/`. The base files contain
`esm_knn`, `structure`, and `active_site`; `.blast.jsonl` and `.hmmer.jsonl`
contain the two homology-search channels.

| Split | ESM-kNN | BLAST | HMMER | Structure | Active site |
|---|---:|---:|---:|---:|---:|
| `30` | 248 | 365 | 376 | 248 | 22 |
| `30-50` | 526 | 551 | 556 | 526 | 55 |
| `price` | 127 | 141 | 143 | 127 | 0 |
| `promiscuous` | 200 | 209 | 209 | 200 | 22 |
| **Total** | **1,101** | **1,266** | **1,284** | **1,101** | **99** |

Every hit was restricted to the training retrieval database. Test queries never
retrieve one another.

## Leakage controls

- Query identifiers and retrieval-neighbor identifiers are pseudonymized in all
  constructed items, evidence files, and predictions.
- Prompts use the sequence and selected evidence, not source-database accessions.
- Raw source accessions are available only as reconstruction inputs; the
  evaluation pipeline does not place them in model prompts as query identifiers.
- `out/items/decontamination.jsonl` provides public IDs and SHA-256 sequence
  digests for overlap checks without restoring accessions.
- The retrieval cache is inductive: only training entries are eligible neighbors.
- Raw API configuration, endpoint addresses, credentials, and experiment logs are
  excluded.

## Known limitations

- The `promiscuous` split is evaluated by single-label any-match accuracy; it is
  not a set-prediction benchmark.
- Difficulty bins do not incorporate evidence availability and therefore are not
  monotonic under open-book evaluation.
- The split names follow upstream conventions; the paper separately reports local
  maximum BLAST identity computed from the cached training hits.
- EC nomenclature changes over time. This release uses the CARE snapshot rather
  than a fresh nomenclature download.
