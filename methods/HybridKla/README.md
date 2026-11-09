# HybridKla

## Paper

- Title: *HybridKla: a hybrid deep learning framework for lactylation site prediction*
- Journal: *Briefings in Bioinformatics* (2025)

## Official Code

- GitHub: [kongxianzw/HybridKla](https://github.com/kongxianzw/HybridKla/)

## File Structure

- `train.py`: feature-branch training, LSTM training, ESM2 fine-tuning, meta-model training, testing, and artifact export
- `predict.py`: standalone prediction from a trained HybridKla checkpoint and side artifacts
- `dataset.py`: TSV loading, sequence normalization, handcrafted feature generation, and LSTM token ids
- `model.py`: DNN/LSTM/ESM2/meta-model definitions and batch prediction helpers
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Branches: ESM2 classifier, LSTM branch, and six handcrafted feature DNN branches
- Handcrafted features: `ACF`, `AAINDEX`, `OBC`, `GPS`, `CKSAAP`, and `PSEAAC`
- Input window length: `seq_len=51`
- Fusion: branch probabilities are stacked and passed to a meta-model
- ESM2 default backbone: [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D)
- Checkpoint format: meta-model checkpoint saved as `model/best_model.pth`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rates: `feature_lr=1.5018e-3`, `lstm_lr=3.1021e-4`, `esm2_lr=1.3638e-4`, `meta_lr=1.0701e-5`
- This implementation does not generate out-of-fold scores with 5-fold cross-validation and does not aggregate 5-fold performance
- GPS uses a fixed encoder fit on the current training split plus the standard BLOSUM62 matrix
- ESM2 weights are downloaded from Hugging Face by default; override with `--esm2_dir` for another Hub model ID or local directory

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
  --esm2_dir facebook/esm2_t30_150M_UR50D \
  --output_dir /path/to/output/hybridkla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. The branch weights under `checkpoint/HybridKla/model/extra/` must be present next to `best_model.pth`.

```bash
python methods/HybridKla/predict.py \
  --checkpoint_path checkpoint/HybridKla/model/best_model.pth \
  --input_tsv dataset/HybridKla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/HybridKla
```

## Output Files

Training output:

- `run.log`
- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `model/best_model.pth`
- `model/extra/lstm.pth`
- `model/extra/lstm_vocab.json`
- `model/extra/meta_feature_order.json`
- `model/extra/gps_encoder.pt`
- `model/extra/esm2_finetuned/`
- `model/extra/esm2_state_dict.pth`
- `model/extra/feature_dnns/*.pth`
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

