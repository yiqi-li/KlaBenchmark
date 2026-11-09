# DeepKla

## Paper

- Title: *DeepKla: predicting lysine lactylation sites in proteins with deep neural networks*

## Official Code

- GitHub: [linDing-group/DeepKla](https://github.com/linDing-group/DeepKla)

## File Structure

- `train.py`: training, validation, testing, checkpoint saving, and metrics export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV loading, sequence normalization, vocabulary, and encoding
- `model.py`: Keras CNN-BiGRU-attention model and model loading
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Input: single amino-acid sequence branch
- Input window length: `seq_len=51`
- Architecture: `Embedding` -> dropout -> `Conv1D` -> `MaxPooling1D` -> `Conv1D` -> `MaxPooling1D` -> bidirectional `GRU` -> attention -> sigmoid classifier
- Optimizer: RMSprop
- Loss: mean squared error, matching the local implementation
- Checkpoint format: Keras model saved as `model/best_model.keras`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rate: `lr=7e-4`
- This implementation does not perform 5-fold cross-validation during training
- Sequences are not padded to length `200`; the default benchmark window is `seq_len=51`
- Model selection uses validation AUROC (`val_auc` in Keras logs, exported as `val_auroc` in result metadata)

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
  --output_dir /path/to/output/deepkla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. `preprocess.json` is read from `checkpoint/DeepKla/`.

```bash
python methods/DeepKla/predict.py \
  --checkpoint_path checkpoint/DeepKla/model/best_model.keras \
  --input_tsv dataset/DeepKla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/DeepKla
```

## Output Files

Training output:

- `run.log`
- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `model/best_model.keras`
- `model/final_model.keras`
- `model/extra/model_summary.txt`
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
