# KlaBenchmark

KlaBenchmark is a benchmark-oriented code and dataset collection for lysine lactylation (Kla) site prediction. It provides standardized datasets and reproducible implementations of seven published Kla predictors:

- PCBert-Kla
- HybridKla
- PBertKla
- DeepKlapred
- ABFF-Kla
- Auto-Kla
- DeepKla

Published Kla predictors were trained and scored on different datasets and background-lysine definitions, so those published numbers cannot be compared with one another. KlaBenchmark keeps the same KlaAtlas sites, splits, and evaluation protocol for all seven methods.

The seven implementations share one training and prediction interface and the partitions reported in Supplementary Table S6. Those resources cover the protein-level and sequence-similarity splits, alternative background definitions, fixed sequence windows, species-specific training, cross-species transfer, and independent DIA tests. Prediction and retraining use `predict.py` and `train.py`. Complete data and weights are on [YiqiLi/KlaBenchmark at Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main).

## Repository Layout

```text
KlaBenchmark/
├── checkpoint/       # Preprocessing metadata; model weights are on Hugging Face
├── dataset/          # Benchmark TSVs grouped as in Supplementary Table S6
├── methods/          # Model code and usage guide
├── report/           # ROC and PRC plotting utility
├── LICENSE
└── README.md
```

## Data and Pretrained Models

Code and the smaller data files are in this repository. The complete benchmark and the trained weights are on [Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main).

| GitHub | Hugging Face |
| --- | --- |
| Source code for the seven predictors | Complete benchmark datasets and pretrained model weights |
| Protein-level split and original-model datasets | The remaining partitions |
| Project documentation | Extra files required for prediction |

On GitHub, each method folder contains `KlaAtlas_benchmark_summary/protein_split_dataset/` and `External_generalization/original_model_dataset/`. Hugging Face adds sequence-similarity, background, window-length, species, cross-species, unseen-site, and DIA partitions under the same seven group names. `checkpoint/` holds the seed-42 protein-level weights. `checkpoint_original/` holds the seed-42 weights trained on each method's original corpus. Place a downloaded package in the matching local folder. PCBert-Kla also needs `model/extra/scaler_model.pkl`, PBertKla needs its encoder files, and HybridKla needs its branch files. See `dataset/README.md` for the directory layout.

## Installation

```bash
pip install -r methods/<method>/requirements.txt
```

See `methods/README.md` for details.

## Quick Start

Download model weights from [YiqiLi/KlaBenchmark on Hugging Face](https://huggingface.co/YiqiLi/KlaBenchmark/tree/main). Place the protein-level weights in `checkpoint/<method>/`, or the original-corpus weights in `checkpoint_original/<method>/`.

`<method>` is one of the seven folder names: `PCBert-Kla`, `HybridKla`, `PBertKla`, `DeepKlapred`, `ABFF-Kla`, `Auto-Kla`, or `DeepKla`. `<checkpoint_file>` depends on the method. Per-method commands and the extra files some predictors require are in [`methods/README.md`](methods/README.md).

```bash
python methods/<method>/predict.py \
  --checkpoint_path checkpoint/<method>/model/<checkpoint_file> \
  --input_tsv <path/to/input.tsv> \
  --output_dir <path/to/output>
```

## Retraining

```bash
python methods/<method>/train.py \
  --train_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/train.tsv \
  --val_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/val.tsv \
  --test_tsv dataset/<method>/KlaAtlas_benchmark_summary/protein_split_dataset/test.tsv \
  --run_name <run_name>
```

## Documentation

- `dataset/README.md`: partition layout and label definitions
- `methods/README.md`: installation, prediction, training, and external model resources
- `report/plot_roc_prc.py`: plot ROC and PRC curves from prediction files

Performance on these partitions is reported in Supplementary Table S6.

## License

The [MIT License](LICENSE) applies to the code in this repository, including `methods/` and `report/`, together with the documentation shipped with that code. Datasets and third-party model weights may be subject to separate terms; see `dataset/README.md` and `methods/README.md`.

## Authors

- Yiqi Li
- Menghuan Zhang

## Citation

Please cite the KlaBenchmark manuscript if you use this repository.
