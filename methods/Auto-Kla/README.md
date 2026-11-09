# Auto-Kla

## Paper

- Title: *Auto-Kla: a novel web server to discriminate lysine lactylation sites using automated machine learning*
- Journal: *Briefings in Bioinformatics* (2023)

## Official Code

- GitHub: [tubic/Auto-Kla](https://github.com/tubic/Auto-Kla)

## File Structure

- `train.py`: training, validation, testing, checkpointing, model soup, and metrics export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV reading, sequence normalization, vocabulary, and encoding
- `model.py`: PyTorch Transformer encoder, classifier, and learning-rate scheduler
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Input: single sequence branch using amino-acid token ids
- Input window length: `seq_len=51`
- Encoder: token embedding + positional embedding + 12-layer PyTorch `TransformerEncoder`
- Classifier: final `[CLS]` hidden state followed by an MLP binary classifier
- Optimizer: `AdamW`
- Learning-rate schedule: warmup plus cosine decay
- Checkpoint format: PyTorch checkpoint saved as `model/best_model.pt`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rate: `lr=3e-4`, `weight_decay=1e-3`, `warmup_ratio=0.1`
- This implementation does not perform 10-fold cross-validation
- The original AutoGluon / Electra discriminator AutoML workflow is not implemented
- The benchmark uses a self-contained PyTorch `TransformerEncoder` implementation for reproducible and stable runs
- Final checkpoint is selected from the top validation AUROC checkpoints using the top-3 model-soup logic implemented in `train.py`

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
  --output_dir /path/to/output/auto_kla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. `preprocess.json` is read from `checkpoint/Auto-Kla/`.

```bash
python methods/Auto-Kla/predict.py \
  --checkpoint_path checkpoint/Auto-Kla/model/best_model.pt \
  --input_tsv dataset/Auto-Kla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/Auto-Kla
```

## Output Files

Training output:

- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `run.log`
- `model/best_model.pt`
- `model/checkpoint/epochXXX_state_dict.pt`
- `model/checkpoint/top3_meta.json`
- `model/extra/best_meta.json`
- `model/extra/best_epoch_state_dict.pt`
- `model/extra/run_meta.json`
- `results/val_metrics.json`
- `results/test_metrics.json`
- `results/test_predictions.tsv` (`RowId`, `Sequence`, `Label`, `Prob`, `PredLabel`)

Prediction output:

- `run.log`
- `predict_config.json`
- `results/test_predictions.tsv` (`RowId`, `Sequence`, `Prob`, `PredLabel`)
- `results/test_metrics.json` when `Label` is available

`Prob` is the predicted probability that the center lysine is a Kla site. `PredLabel` is 1 when `Prob` is at least 0.5.
