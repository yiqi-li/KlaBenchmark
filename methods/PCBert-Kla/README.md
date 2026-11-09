# PCBert-Kla

## Paper

- Title: *PCBert-Kla: an efficient prediction method for lysine lactylation sites based on ProtBert and fusion of physicochemical features*

## Official Code

- GitHub: [ZhangHongqi215/PCBert-Kla](https://github.com/ZhangHongqi215/PCBert-Kla)

## File Structure

- `train.py`: training, validation, testing, checkpoint saving, and metrics export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV loading, sequence normalization, physicochemical feature construction, dataset, and collate function
- `model.py`: PCBert-Kla classifier, ProtBert loading, and BERT-layer truncation helper
- `utils.py`: logging, metrics, JSON helpers, run paths, scaler saving, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Input branches: ProtBert sequence representation and physicochemical feature branch
- Input window length: `seq_len=51`
- ProtBert CLS embedding dimension: `1024`
- Physicochemical feature dimension: `27`
- Fusion dimension: `1051`
- Classifier: attention on the fused 1051-dimensional feature vector followed by MLP (`1051 -> 32 -> 8 -> 1`)
- Default ProtBert backbone: [Rostlab/prot_bert_bfd](https://huggingface.co/Rostlab/prot_bert_bfd)
- Checkpoint format: PyTorch checkpoint saved as `model/best_model.pt`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rates: `lr_bert=2e-5`, `lr_other=1e-3`
- Optimizer: `AdamW`
- Training does not use 5-fold cross-validation
- Attention is applied to the fused 1051-dimensional input feature vector, matching the paper description more closely
- The physicochemical feature scaler is fit on the current training split and saved as `model/extra/scaler_model.pkl`
- ProtBert weights are downloaded from Hugging Face by default; use `--local_files_only` only for cached/offline runs

## Input Format

`Sequence` is a 51 aa window centered on a lysine. `_` pads positions outside the protein.

- Training TSV: `Sequence` and `Label` (`1` for a Kla site, `0` for a background lysine).
- Evaluation TSV: the same two columns. `Label` is used to compute metrics.
- Prediction TSV: `Sequence` is enough. `Label` is optional.

Benchmark files also contain `Site`, `ACC_ID`, `Mod_positions`, `Species`, and `UniProtTaxID`. Those columns are copied through to the prediction table and are not required for a new input.

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
  --protbert_dir Rostlab/prot_bert_bfd \
  --output_dir /path/to/output/pcbert_kla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root after placing the Hugging Face checkpoint at `checkpoint/PCBert-Kla/`. `--input_tsv` and `--test_tsv` are the same option. The scaler is read from `model/extra/scaler_model.pkl`.

```bash
python methods/PCBert-Kla/predict.py \
  --checkpoint_path checkpoint/PCBert-Kla/model/best_model.pt \
  --input_tsv dataset/PCBert-Kla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/PCBert-Kla
```

## Output Files

Training output:

- `run.log`
- `run_config.json`
- `preprocess.json`
- `dataset_stats.json`
- `training_log.csv`
- `model/best_model.pt`
- `model/extra/scaler_model.pkl`
- `model/extra/run_meta.json`
- `model/extra/protbert_finetuned/` when `save_pretrained=true`
- `results/val_metrics.json`
- `results/test_metrics.json`
- `results/test_predictions.tsv` (`RowId`, `Sequence`, `Label`, `Prob`, `PredLabel`)

Prediction output:

- `run.log`
- `predict_config.json`
- `predictions.tsv` (`RowId`, `Sequence`, `Prob`, `PredLabel`)
- `predict_metrics.json` when `Label` is available

`Prob` is the predicted probability that the center lysine is a Kla site. `PredLabel` is 1 when `Prob` is at least 0.5.
