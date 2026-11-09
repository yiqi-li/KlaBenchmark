# KlaBenchmark Datasets

Benchmark TSV files for Kla site prediction are released at [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). This directory contains the protein-level split and the original-model datasets. The other partitions are in the Hugging Face release.

The files under each method are grouped in the same order as Supplementary Table S6.

## Download Full Dataset

The complete benchmark dataset is available from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). Download the full dataset archive, then unpack or copy each method-specific folder into this local `dataset/` directory.

Expected local layout:

```text
dataset/
├── ABFF-Kla/
├── Auto-Kla/
├── DeepKla/
├── DeepKlapred/
├── HybridKla/
├── PBertKla/
└── PCBert-Kla/
```

Each method folder contains these groups:

```text
dataset/<method>/
├── KlaAtlas_benchmark_summary/
│   ├── protein_split_dataset/
│   ├── sequence_split_dataset/
│   └── unseen_external/
├── External_generalization/
│   └── original_model_dataset/
├── Background_definition/
│   ├── background_ms_observable/
│   ├── background_protein_level/
│   ├── background_rsa_lt_0.2/
│   └── background_bias_removed/
├── Sequence_context/
│   ├── 15aa_window_protein_split_dataset/
│   ├── 31aa_window_protein_split_dataset/
│   ├── 41aa_window_protein_split_dataset/
│   └── 51aa_window_protein_split_dataset/
├── Cross_species_transfer/
│   └── cross_species_dataset/
├── Species_specific_training/
│   └── species_dataset/
└── DIA_external_performance/
    └── dia_external/
```

After downloading, pass the corresponding TSV paths to each method:

```bash
python methods/<method>/train.py \
  --train_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/train.tsv \
  --val_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/val.tsv \
  --test_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv
```

Keep the [Hugging Face release](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main) directory structure intact when possible. The group names and the dataset folder names inside them are part of the benchmark paths.

## Required Columns

For most predictors, each `train.tsv`, `val.tsv`, and `test.tsv` must include at least:

- **`Sequence`**: local sequence window centered on the candidate lysine. Padding is `_`.
- **`Label`**: `1` for an experimentally identified Kla site, `0` for a background lysine site

**ABFF-Kla** is an exception: inputs also need structure-related fields such as **`Contact`**. See `methods/ABFF-Kla/README.md`.

## Optional Columns

Depending on the split, files may also include:

- `Site`: site-level identifier
- `ACC_ID`: protein accession (e.g. UniProt)
- `Mod_positions`: lysine position in the protein
- `Species`: source species
- `UniProtTaxID`: NCBI taxonomy identifier

## Benchmark Partitions

### `KlaAtlas_benchmark_summary/`

| Directory | Benchmark setting |
| --- | --- |
| `protein_split_dataset/` | Protein-level train/validation/test split. Samples from the same protein stay in one subset. |
| `sequence_split_dataset/` | The same KlaAtlas sites, repartitioned after clustering lysine-centered windows at 50% identity (CD-HIT). |
| `unseen_external/` | Class-balanced unseen-site external test. Ten repeats, `test.tsv` only. Sites absent from the published training sets are held out from the KlaAtlas protein-level training and validation proteins. |

### `External_generalization/`

| Directory | Benchmark setting |
| --- | --- |
| `original_model_dataset/` | Train, validation, and test files aligned with each published model's own corpus. Direct external testing uses the KlaAtlas protein-level test in `KlaAtlas_benchmark_summary/protein_split_dataset/`. |

### `Background_definition/`

Each background is a separate dataset with `repeat_01` through `repeat_10`. Every repeat has `train.tsv`, `val.tsv`, and `test.tsv`.

| Directory | Benchmark setting |
| --- | --- |
| `background_ms_observable/` | Lysines covered by experimentally identified peptides and lacking Kla evidence. |
| `background_protein_level/` | Lysines on Kla-containing proteins that were not identified as Kla sites, irrespective of peptide-level MS observability. |
| `background_rsa_lt_0.2/` | MS-observable lysines with RSA < 0.2. |
| `background_bias_removed/` | MS-observable background after the peptide-construction bias is removed. |

### `Sequence_context/`

These datasets use the protein-level partition. Only the lysine-centered window length changes.

| Directory | Benchmark setting |
| --- | --- |
| `15aa_window_protein_split_dataset/` | 15 aa window |
| `31aa_window_protein_split_dataset/` | 31 aa window |
| `41aa_window_protein_split_dataset/` | 41 aa window |
| `51aa_window_protein_split_dataset/` | 51 aa window |

### `Cross_species_transfer/`

| Directory | Benchmark setting |
| --- | --- |
| `cross_species_dataset/` | Train and validate on human proteins. Non-human species are used only in the external test. |

### `Species_specific_training/`

| Directory | Benchmark setting |
| --- | --- |
| `species_dataset/<Species>/` | Protein-level split built from that species alone. |

`species_dataset/` contains `Homo_sapiens`, `Mus_musculus`, `Rattus_norvegicus`, `Caenorhabditis_elegans`, `Escherichia_coli`, and `Streptococcus_mutans`.

### `DIA_external_performance/`

| Directory | Benchmark setting |
| --- | --- |
| `dia_external/repeat_01/` ... `repeat_10/` | Independent DIA evaluation. These files are not used for training or model selection. |

Each repeat contains `LAC293-Kla.tsv`, `PGH-Kla.tsv`, `RAT-Kla.tsv`, and `RIH-Kla.tsv`.

Most other partitions provide:

```text
train.tsv
val.tsv
test.tsv
```

### Selecting paths by scenario

| Scenario | Typical data paths |
| --- | --- |
| Protein-level split | `KlaAtlas_benchmark_summary/protein_split_dataset/train.tsv`, etc. |
| Sequence-level split | `KlaAtlas_benchmark_summary/sequence_split_dataset/train.tsv`, etc. |
| Unseen-site external test | `KlaAtlas_benchmark_summary/unseen_external/repeat_01/test.tsv` |
| Original-model setting | `External_generalization/original_model_dataset/train.tsv`, etc. |
| Background comparison | `Background_definition/background_ms_observable/repeat_01/train.tsv`, and the matching repeat under the other three background folders |
| Fixed window (15 / 31 / 41 / 51 aa) | `Sequence_context/15aa_window_protein_split_dataset/...`, and the 31, 41, and 51 aa folders |
| Cross-species transfer | `Cross_species_transfer/cross_species_dataset/train.tsv`, etc. |
| Species-specific training | `Species_specific_training/species_dataset/<Species>/train.tsv`, etc. |
| DIA external evaluation | `DIA_external_performance/dia_external/repeat_01/LAC293-Kla.tsv`, and the other three DIA files in that repeat |

For fixed-window partitions, use the window length that matches the dataset name when configuring the model (e.g. `--seq_len` where supported). Report window length in published results; model rankings can change with context size.

## Label Caveat

Background lysine sites (`Label = 0`) are **not** experimentally verified negatives. They are lysine residues that were not identified as Kla sites under the rule used by that partition. Treat benchmark tasks as **positive versus background** classification, not positive versus confirmed-negative.

## Usage Guidelines

- Keep official train/validation/test splits unchanged when reproducing published results.
- State which partition (protein-level, sequence-level, fixed window, species, cross-species, background, unseen-site, DIA, or original-model) was used.
- Document any extra filtering, species subset, or custom window length.
- Performance for these partitions is reported in Supplementary Table S6. Background, unseen-site, and DIA results are means over the ten released repeats.

## Redistribution

Benchmark data are for research use. When redistributing derivatives, comply with the terms of the original proteomics databases, UniProt, and source publications, and cite the KlaBenchmark manuscript and underlying studies.
