from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

METHOD_NAME = "PCBert-Kla"
OFFICIAL_KEEP_BERT_LAYERS = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "PCBert-Kla" / "KlaAtlas_benchmark_summary" / "protein_split_dataset"
DEFAULT_TRAIN_TSV = str(DEFAULT_DATA_DIR / "train.tsv")
DEFAULT_VAL_TSV = str(DEFAULT_DATA_DIR / "val.tsv")
DEFAULT_TEST_TSV = str(DEFAULT_DATA_DIR / "test.tsv")
DEFAULT_PROTBERT_DIR = "Rostlab/prot_bert_bfd"
DEFAULT_OUT_ROOT = str(Path(__file__).resolve().parent / "runs")


@dataclass
class Config:
    train_tsv: str = DEFAULT_TRAIN_TSV
    val_tsv: str = DEFAULT_VAL_TSV
    test_tsv: str = DEFAULT_TEST_TSV
    output_dir: str = ""
    out_root: str = DEFAULT_OUT_ROOT
    run_name: str = ""
    checkpoint_path: str = ""
    scaler_path: str = ""
    run_meta_path: str = ""
    protbert_dir: str = DEFAULT_PROTBERT_DIR
    keep_bert_layers: int = OFFICIAL_KEEP_BERT_LAYERS
    local_files_only: bool = False
    save_pretrained: bool = True
    device: str = "cuda"
    gpu: str = ""
    seed: int = 42
    seq_len: int = 51
    batch_size: int = 512
    epochs: int = 100
    lr_bert: float = 2e-5
    lr_other: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5
    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0
    decision_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_field_names() -> set[str]:
    return {field.name for field in fields(Config)}


def load_config_file(path: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    valid_keys = _config_field_names()
    unknown_keys = sorted(set(data) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    return data


def merge_config(base: Config, *overrides: dict[str, Any]) -> Config:
    merged = base.to_dict()
    valid_keys = _config_field_names()
    for override in overrides:
        for key, value in override.items():
            if value is None or key not in valid_keys:
                continue
            merged[key] = value
    return Config(**merged)


def make_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate PCBert-Kla.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--val_tsv", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--output_dir", default=None, help="Resolved run directory. Overrides --out_root/--run_name.")
    parser.add_argument("--out_root", default=None, help="Root directory used when --output_dir is not provided.")
    parser.add_argument("--run_name", default=None, help="Run subdirectory name under --out_root.")

    parser.add_argument("--protbert_dir", "--model_name", dest="protbert_dir", default=None)
    parser.add_argument("--keep_bert_layers", type=int, default=None)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load ProtBert from local files only.",
    )
    parser.add_argument(
        "--save_pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save the fine-tuned ProtBert weights with save_pretrained().",
    )

    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None)
    parser.add_argument("--lr_bert", type=float, default=None)
    parser.add_argument("--lr_other", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def make_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained PCBert-Kla checkpoint and run prediction.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--checkpoint_path", default=None, help="Path to the saved best_model.pt checkpoint.")
    parser.add_argument("--test_tsv", "--input_tsv", dest="test_tsv", default=None, help="Input TSV for prediction.")
    parser.add_argument("--output_dir", default=None, help="Directory for prediction outputs.")
    parser.add_argument("--scaler_path", default=None, help="Optional override for scaler_model.pkl.")
    parser.add_argument("--run_meta_path", default=None, help="Optional override for run_meta.json.")

    parser.add_argument("--protbert_dir", "--model_name", dest="protbert_dir", default=None)
    parser.add_argument("--keep_bert_layers", type=int, default=None)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load ProtBert from local files only.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def config_and_overrides_from_args(args: argparse.Namespace) -> tuple[Config, dict[str, Any], dict[str, Any]]:
    args_dict = vars(args).copy()
    config_path = args_dict.pop("config", "")
    file_overrides = load_config_file(config_path)
    cli_overrides = {key: value for key, value in args_dict.items() if value is not None}
    config = merge_config(Config(), file_overrides, cli_overrides)
    return config, file_overrides, cli_overrides
