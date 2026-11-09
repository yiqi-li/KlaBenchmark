# ABFF-Kla

## Paper

- Title: *Lactylation prediction models based on protein sequence and structural feature fusion*

## Official Code

- GitHub: [ispotato/Lactylation_model](https://github.com/ispotato/Lactylation_model)

## File Structure

- `train.py`: training, validation, testing, checkpoint saving, and metrics export
- `predict.py`: standalone prediction from a trained checkpoint
- `dataset.py`: TSV loading, sequence/contact encoding, and contact-segment generation
- `model.py`: ABFF-Kla model, self-attention layer, branch pretraining, and model loading
- `utils.py`: logging, metrics, JSON helpers, run paths, and prediction export
- `configs.py`: default hyperparameters, dataclass config, CLI, and JSON config loading

## Model Summary

- Input branches: sequence context branch (`Context` or `Sequence`) and structure-derived contact branch (`Contact`)
- Input window length: `seq_len=35`
- Encoding: `_` is padding id `0`; canonical amino acids are `1..20`; unknown residues are `21`
- Branch architecture: `Embedding` -> self-attention -> `Conv1D` -> `LSTM`
- Fusion head: concatenate branch features -> fusion attention -> dense MLP -> sigmoid score
- Training flow: pretrain acid branch, pretrain contact branch, then train the fusion model
- Checkpoint format: Keras model saved as `model/best_model.keras`

## Reproduction Notes

- Unified benchmark defaults: `batch_size=512`, `epochs=100`, early stopping on validation AUROC, patience `5`
- Learning rate: `lr=7e-4`
- This implementation does not perform multiple random splits plus 10-fold cross-validation
- If `Contact` is missing, contact segments are generated from local AlphaFold PDB files specified by `--pdb_dir`

## Input Format

ABFF-Kla uses two input branches: a sequence context branch and a structure-derived contact branch. Both branches are required at runtime.

Training, validation, and test TSV files should contain:

- `Context` or `Sequence`: sequence window centered on the modification site
- `Contact`: structure-derived contact window centered on the modified lysine
- `Label`

### Contact

`Contact` is a fixed-length amino acid segment (`seq_len`, default `35`) used by the contact branch. It is built from residues whose Cα atoms lie within **10 Å** of the modification site (default cutoff `contact_cutoff=10.0`). The modified lysine is kept at the center of the window; nearby contact residues are selected from the AlphaFold structure and padded with `_` when needed.

You can provide `Contact` directly in the TSV, or let `train.py` / `predict.py` generate it automatically.

### Generating Contact from AlphaFold PDBs

If `Contact` is missing, the TSV must also contain:

- `ACC_ID`: protein accession used to locate the AlphaFold PDB
- `Mod_positions`: 1-based lysine modification position

In that case, pass `--pdb_dir` pointing to a directory of **AlphaFold-predicted protein structures** (for example `AF-{ACC_ID}-F1-model.pdb`). `ensure_with_contact_tsv()` reads each PDB, finds residues within 10 Å of the site, and writes a derived `*.with_contact.tsv` under the run cache directory.

Example:

```bash
python train.py \
  --train_tsv /path/to/train.tsv \
  --pdb_dir /path/to/alphafold_pdb
```

### Prediction input

For a new prediction, provide `Sequence` or `Context` as a 35 aa window centered on the lysine. The released benchmark files already contain `Contact`, so `--pdb_dir` is not needed for those files. If `Contact` is absent, also provide `ACC_ID` and `Mod_positions`, and pass `--pdb_dir`. `Label` is optional and is used only to compute metrics.

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
  --pdb_dir /path/to/alphafold_pdb \
  --output_dir /path/to/output/abff_kla_run
```

JSON configs are also supported:

```bash
python train.py --config /path/to/train_config.json
```

## Prediction

Run this from the repository root. The protein-level test file in this repository already contains `Contact`.

```bash
python methods/ABFF-Kla/predict.py \
  --checkpoint_path checkpoint/ABFF-Kla/model/best_model.keras \
  --input_tsv dataset/ABFF-Kla/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --output_dir results/ABFF-Kla
```

Add `--pdb_dir` only when the input has no `Contact` column.

## Output Files

Training output:

- `run_config.json`
- `dataset_stats.json`
- `preprocess.json`
- `training_log.csv`
- `run.log`
- `model/best_model.keras`
- `model/final_model.keras`
- `model/extra/acid0.h5`
- `model/extra/contmap0.h5`
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
