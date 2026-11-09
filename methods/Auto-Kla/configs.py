from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

METHOD_NAME = "Auto-Kla"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "Auto-Kla" / "KlaAtlas_benchmark_summary" / "protein_split_dataset"
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
    top3_meta_path: str = ""
    seed: int = 42
    gpu: str = ""
    device: str = "cuda"
    seq_len: int = 51
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_ratio: float = 0.1
    patience: int = 5
    d_model: int = 256
    n_heads: int = 8
    dim_ff: int = 1024
    dropout: float = 0.1
    num_workers: int = 2
    grad_clip: float = 1.0
    num_layers: int = 12
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
    parser = argparse.ArgumentParser(description="Train and evaluate Auto-Kla.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--val_tsv", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dim_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def make_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained Auto-Kla checkpoint and run prediction.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--checkpoint_path", default=None, help="Path to the saved best_model.pt checkpoint.")
    parser.add_argument("--test_tsv", "--input_tsv", dest="test_tsv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--preprocess_path", default=None)
    parser.add_argument("--run_meta_path", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--device", default=None)
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
