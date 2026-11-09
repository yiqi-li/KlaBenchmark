# DeepKlapred

## Paper

- Title: *DeepKlapred: A deep learning framework for identifying protein lysine lactylation sites via multi-view feature fusion*

## Official Code

- GitHub: [GGCL7/DeepKlapred](https://github.com/GGCL7/DeepKlapred)

## File Structure

- `train.py`: training, validation, testing, checkpoint saving, and metrics export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV loading, sequence normalization, descriptor generation, and dataset class
- `model.py`: sequence branch, descriptor branch, transformer blocks, cross-attention fusion, and classifier
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Input branches: sequence embedding branch and handcrafted descriptor branch
- Input window length: `seq_len=51`
- Sequence token length: `seq_len + 1` because `[CLS]` is prepended
- Sequence branch: residue embedding -> sinusoidal positional encoding -> BiGRU -> Transformer blocks
- Descriptor branch: 739-dimensional handcrafted descriptor vector
- Fusion: cross-attention and MLP classifier
- Checkpoint format: PyTorch checkpoint saved as `model/best_model.pt`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rate: `lr=3e-3`
- GRU / Transformer dropout is `0.2`
- Positional encoding follows the official implementation with sinusoidal encoding rather than learnable positional embeddings
- Descriptor columns for validation and test are reindexed to the training feature columns for deterministic prediction

## Input Format

`Sequence` is a 51 aa window centered on a lysine. `_` pads positions outside the protein.

- Training TSV: `Sequence` and `Label` (`1` for a Kla site, `0` for a background lysine).
- Evaluation TSV: the same two columns. `Label` is used to compute metrics.
- Prediction TSV: `Sequence` is enough. `Label` is optional.

## Training

Use default paths from `configs.py`:

```bash
python train.py
```

Or provide explicit paths:

```bash
python train.py \
  --train_tsv /path/to/train.tsv \
  --val_tsv /path/to/val.tsv \
  --test_tsv /path/to/test.tsv \
  --output_dir /path/to/output/deepklapred_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. `preprocess.json` is read from `checkpoint/DeepKlapred/`.

```bash
python methods/DeepKlapred/predict.py \
  --checkpoint_path checkpoint/DeepKlapred/model/best_model.pt \
  --input_tsv dataset/DeepKlapred/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/DeepKlapred
```

## Output Files

Training output:

- `run.log`
- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `model/best_model.pt`
- `model/extra/run_meta.json`
- `results/val_metrics.json`
- `results/test_metrics.json`
- `results/test_predictions.tsv` (`RowId`, `Sequence`, `Label`, `Prob`, `PredLabel`)

Prediction output:

- `run.log`
- `predict_config.json`
- `predictions.tsv` (`RowId`, `Sequence`, `Prob`, `PredLabel`)
- `predict_metrics.json` when `Label` is available

`Prob` is the predicted probability that the center lysine is a Kla site. `PredLabel` is 1 when `Prob` is at least 0.5.
