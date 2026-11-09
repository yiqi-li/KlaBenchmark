# PBertKla

## Paper

- Title: *PBertKla: a protein large language model for predicting human lysine lactylation sites*
- Journal: *BMC Biology* (2025)

## Official Code

- GitHub: [laihongyan/PBertKla](https://github.com/laihongyan/PBertKla)
- ProteinBERT upstream: [nadavbra/protein_bert](https://github.com/nadavbra/protein_bert)

## File Structure

- `train.py`: three-stage fine-tuning, validation, testing, checkpoint selection, and artifact export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV loading, sequence validation, and ProteinBERT-compatible preprocessing
- `model.py`: TensorFlow setup, ProteinBERT loading, model construction, and encoder artifact helpers
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Backbone: ProteinBERT
- Framework: TensorFlow / Keras
- Input: sequence branch only
- Default training sequence length: `seq_len=45`
- Final-stage sequence length: `final_seq_len=1024`
- Classifier dropout: `0.5`
- Pretrained weights: [Zakia/ProteinBERT](https://huggingface.co/Zakia/ProteinBERT), file `epoch_92400_sample_23500000.pkl`
- Checkpoint format: Keras weights saved as `model/best_model.weights.h5`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rates: `lr_stage1=2e-3`, `lr_stage2=2e-3`, `lr_stage3=1e-4`
- The final best model may come from `stage1_frozen`, `stage2_unfrozen`, or `stage3_final`
- The first two stages are not padded to length `256`; they use `seq_len=45`
- `<START>` and `<END>` are added by ProteinBERT and are not counted in `seq_len`
- ProteinBERT weights are downloaded from Hugging Face by default; `--pretrained_cache_dir` is only a local override

## Input Format

`Sequence` is a 45 aa window centered on a lysine. `_` pads positions outside the protein. ProteinBERT adds its own start and end tokens and does not count them in `seq_len`.

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
  --output_dir /path/to/output/pbertkla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. Place `model/extra/input_encoder.pkl` and `model/extra/output_spec.pkl` next to the checkpoint. The ProteinBERT backbone is downloaded from [Zakia/ProteinBERT](https://huggingface.co/Zakia/ProteinBERT) unless `--pretrained_cache_dir` points to a local copy.

```bash
python methods/PBertKla/predict.py \
  --checkpoint_path checkpoint/PBertKla/model/best_model.weights.h5 \
  --input_tsv dataset/PBertKla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/PBertKla
```

## Output Files

Training output:

- `run.log`
- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `model/best_model.weights.h5`
- `model/checkpoint/best_stage1_frozen_seq45.weights.h5`
- `model/checkpoint/best_stage2_unfrozen_seq45.weights.h5`
- `model/checkpoint/best_stage3_final_seq1024.weights.h5` when enabled
- `model/extra/input_encoder.pkl`
- `model/extra/output_spec.pkl`
- `model/extra/run_meta.json`
- `model/extra/saved_model/` when TensorFlow SavedModel export succeeds
- `model/extra/model_architecture.json` or `model/extra/model_summary.txt`
- `results/val_metrics.json`
- `results/test_metrics.json`
- `results/curves.json`
- `results/test_predictions.tsv` (`RowId`, `Sequence`, `Label`, `Prob`, `PredLabel`)

Prediction output:

- `run.log`
- `predict_config.json`
- `predictions.tsv` (`RowId`, `Sequence`, `Prob`, `PredLabel`)
- `predict_metrics.json` when `Label` is available

`Prob` is the predicted probability that the center lysine is a Kla site. `PredLabel` is 1 when `Prob` is at least 0.5.
