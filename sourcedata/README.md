# Pre-construction Source Data

This directory contains the original sequence tables and EC snapshot used before
construction of EC-Reason-Bench. The files are retained in their upstream
CARE/PoinnCARE layout so the benchmark items can be rebuilt without a separate
data download.

## Inventory

`data/all_ec.txt` contains the 4,963 valid EC leaves used to construct the
hierarchy.

`data/care_graph/` contains:

- `30_protein_test.csv`, `30-50_protein_test.csv`,
  `price_protein_test.csv`, and `promiscuous_protein_test.csv`: the four
  original test splits;
- `protein_train50.csv`: the original training and retrieval sequence table.

Generated intermediates are intentionally not bundled: `entry2index.pkl`,
`ec2idx.pkl`, `feats.npy`, `labels.npy`, and the Foldseek/Folddisco sparse
graphs. The included constructed evidence cache makes them unnecessary for
evaluation and scoring.

The constructed benchmark is stored separately under `out/`: `out/items/`
contains model-ready benchmark records, `out/evidence/` contains the offline
retrieval cache, and `out/results/` contains structured predictions and metrics.

## Identifier and privacy policy

Raw source files retain upstream protein accessions so the sequence tables remain
auditable and the benchmark items can be reconstructed. Construction replaces
query and retrieval-neighbor accessions with deterministic pseudonyms in the
public artifacts. Evaluation prompts use sequences and selected evidence rather
than raw query accessions.

This snapshot contains no API credentials, private endpoints, experiment logs,
or local filesystem paths. `download.log` is intentionally excluded because it
is a transfer log, not benchmark data.

No new license is asserted over third-party records. Review the upstream CARE,
PoinnCARE, and source-database terms before further redistribution.
