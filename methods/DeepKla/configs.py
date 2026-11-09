from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

METHOD_NAME = "DeepKla"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "DeepKla" / "KlaAtlas_benchmark_summary" / "protein_split_dataset"
DEFAULT_TRAIN_TSV = str(DEFAULT_DATA_DIR / "train.tsv")
DEFAULT_VAL_TSV = str(DEFAULT_DATA_DIR / "val.tsv")
DEFAULT_TEST_TSV = str(DEFAULT_DATA_DIR / "test.tsv")
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
    preprocess_path: str = ""
    run_meta_path: str = ""
    seed: int = 42
    gpu: str = ""
    seq_len: int = 51
    batch_size: int = 512
    epochs: int = 100
    lr: float = 7e-4
    patience: int = 5
    max_train: int = 0
    max_val: int = 0
    max_test: int = 0
    decision_threshold: float = 0.5

    # model-specific parameters
    embedding_dim: int = 128
    conv_filters: int = 64
    gru_units: int = 50
    dropout: float = 0.25

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
    parser = argparse.ArgumentParser(description="Train and evaluate DeepKla.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--val_tsv", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-train", "--max_train", dest="max_train", type=int, default=None)
    parser.add_argument("--max-val", "--max_val", dest="max_val", type=int, default=None)
    parser.add_argument("--max-test", "--max_test", dest="max_test", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def make_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained DeepKla checkpoint and run prediction.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--checkpoint_path", default=None, help="Path to the saved best_model.keras checkpoint.")
    parser.add_argument("--test_tsv", "--input_tsv", dest="test_tsv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--preprocess_path", default=None)
    parser.add_argument("--run_meta_path", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    return parser


def config_and_overrides_from_args(args: argparse.Namespace) -> tuple[Config, dict[str, Any], dict[str, Any]]:
    args_dict = vars(args).copy()
    config_path = args_dict.pop("config", "")
    file_overrides = load_config_file(config_path)
    cli_overrides = {key: value for key, value in args_dict.items() if value is not None}
    config = merge_config(Config(), file_overrides, cli_overrides)
    return config, file_overrides, cli_overrides
