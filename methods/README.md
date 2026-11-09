# KlaBenchmark Methods

This directory contains reproducible implementations of seven published Kla site predictors. Each implementation lives under `methods/<method>/` with a shared layout: `configs.py`, `dataset.py`, `model.py`, `train.py`, `predict.py`, and `requirements.txt`. Per-model CLI details are in `methods/<method>/README.md`.

Supported predictors: ABFF-Kla, Auto-Kla, DeepKla, DeepKlapred, HybridKla, PBertKla, PCBert-Kla.

## Installation

```bash
pip install -r methods/<method>/requirements.txt
```

Replace `<method>` with the target folder name (e.g. `Auto-Kla` → `methods/Auto-Kla/requirements.txt`).

Use a dedicated virtual environment per run if you switch between implementations with incompatible dependencies.

## Quick Start (Prediction)

1. Install dependencies (above).
2. Download the `checkpoint` archive for your model from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main).
3. Place the downloaded weights and metadata under the corresponding local directory, for example `checkpoint/<method>/`.
4. Prepare an input TSV with the columns required by that method (see `dataset/README.md`).
5. Run prediction:

```bash
python methods/<method>/predict.py \
  --checkpoint_path checkpoint/<method>/model/<checkpoint_file> \
  --input_tsv <path/to/input.tsv> \
  --output_dir <path/to/output>
```

Predictions and optional metrics are written under `--output_dir`. Checkpoint file extensions differ by implementation (`.pt`, `.keras`, `.weights.h5`, etc.); use the artifact named in the [YiqiLi/KlaBenchmark Hugging Face model card](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main).

## Pretrained Checkpoints

Two checkpoint packages are available from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). `checkpoint/` contains the seed-42 protein-level model. `checkpoint_original/` contains the seed-42 model trained on `External_generalization/original_model_dataset`. Download the package you need, then unpack it into the matching local folder:

```text
checkpoint/
├── ABFF-Kla/
├── Auto-Kla/
├── DeepKla/
├── DeepKlapred/
├── HybridKla/
├── PBertKla/
└── PCBert-Kla/
```

Keep the downloaded package structure intact when possible, because `predict.py` can infer side artifacts from the checkpoint run directory. Typical files include:

```text
checkpoint/<method>/
├── preprocess.json
├── run_config.json
├── model/
│   ├── best_model.<ext>
│   └── extra/
│       └── run_meta.json
└── results/
```

Some methods require additional files in `model/extra/` for prediction, such as `scaler_model.pkl` for PCBert-Kla, ProteinBERT encoder artifacts for PBertKla, or branch models for HybridKla. These files should be downloaded together with the checkpoint package from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main) and placed in the same relative locations.

## Retraining

To reproduce or extend a benchmark, install dependencies, download the benchmark TSV splits from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main) (see `dataset/README.md`), then train:

```bash
python methods/<method>/train.py \
  --train_tsv <path/to/train.tsv> \
  --val_tsv <path/to/val.tsv> \
  --test_tsv <path/to/test.tsv> \
  --run_name <run_name>
```

Defaults in each `configs.py` point at repository-relative paths when local data exist; override paths via CLI or `--config <file.json>`.

Training outputs are written to `methods/<method>/runs/<run_name>/` (checkpoints, logs, `test_metrics.json`, `test_predictions.tsv`). These paths are ignored by Git.

Choose train/validation/test TSV paths according to the benchmark partition described in `dataset/README.md`.

## External Resources

Large pretrained backbones are **not** stored in Git. By default, methods that depend on them download weights from the [Hugging Face Hub](https://huggingface.co) on first use (cached under your Hugging Face home directory). You can override with a local path or another Hub model ID via CLI flags.

| Method | Default Hugging Face model | CLI option |
| --- | --- | --- |
| PCBert-Kla | [Rostlab/prot_bert_bfd](https://huggingface.co/Rostlab/prot_bert_bfd) | `--protbert_dir` |
| HybridKla | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) | `--esm2_dir` |
| PBertKla | [Zakia/ProteinBERT](https://huggingface.co/Zakia/ProteinBERT) (`epoch_92400_sample_23500000.pkl`) | `--pretrained_hf_repo`, `--pretrained_hf_filename` |

For **ABFF-Kla**, AlphaFold PDB structures are still provided as local files (see `--pdb_dir` in `methods/ABFF-Kla/README.md`).

| Override | CLI option |
| --- | --- |
| Local ProteinBERT dump directory (skip Hub) | `--pretrained_cache_dir` |
| ProtBert: force offline/local cache only | `--local_files_only` |
| AlphaFold PDB directory | `--pdb_dir` |



## Method Layout

```text
methods/<method>/
├── configs.py
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```
