from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

METHOD_NAME = "PBertKla"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "PBertKla" / "KlaAtlas_benchmark_summary" / "protein_split_dataset"
DEFAULT_TRAIN_TSV = str(DEFAULT_DATA_DIR / "train.tsv")
DEFAULT_VAL_TSV = str(DEFAULT_DATA_DIR / "val.tsv")
DEFAULT_TEST_TSV = str(DEFAULT_DATA_DIR / "test.tsv")
DEFAULT_PRETRAINED_CACHE_DIR = ""
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
    run_meta_path: str = ""
    input_encoder_path: str = ""
    output_spec_path: str = ""
    pretrained_cache_dir: str = DEFAULT_PRETRAINED_CACHE_DIR
    gpu: str = ""
    seed: int = 42
    lr_stage1: float = 2e-3
    lr_stage2: float = 2e-3
    lr_stage3: float = 1e-4
    batch_size: int = 512
    seq_len: int = 45
    dropout: float = 0.5
    epochs: int = 100
    final_seq_len: int = 1024
    n_final_epochs: int = 1
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
    parser = argparse.ArgumentParser(description="Train and evaluate PBertKla.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--val_tsv", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--output_dir", "--out", dest="output_dir", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr-stage1", "--lr_stage1", dest="lr_stage1", type=float, default=None)
    parser.add_argument("--lr-stage2", "--lr_stage2", dest="lr_stage2", type=float, default=None)
    parser.add_argument("--lr-stage3", "--lr_stage3", dest="lr_stage3", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--final-seq-len", "--final_seq_len", dest="final_seq_len", type=int, default=None)
    parser.add_argument("--n-final-epochs", "--n_final_epochs", dest="n_final_epochs", type=int, default=None)
    parser.add_argument("--pretrained-cache-dir", "--pretrained_cache_dir", dest="pretrained_cache_dir", default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def make_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained PBertKla checkpoint and run prediction.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--checkpoint_path", default=None, help="Path to the saved best_model.weights.h5 checkpoint.")
    parser.add_argument("--test_tsv", "--input_tsv", dest="test_tsv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_meta_path", default=None)
    parser.add_argument("--input_encoder_path", default=None)
    parser.add_argument("--output_spec_path", default=None)
    parser.add_argument("--pretrained-cache-dir", "--pretrained_cache_dir", dest="pretrained_cache_dir", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    return parser


def config_and_overrides_from_args(args: argparse.Namespace) -> tuple[Config, dict[str, Any], dict[str, Any]]:
    args_dict = vars(args).copy()
    config_path = args_dict.pop("config", "")
    file_overrides = load_config_file(config_path)
    cli_overrides = {key: value for key, value in args_dict.items() if value is not None}
    config = merge_config(Config(), file_overrides, cli_overrides)
    return config, file_overrides, cli_overrides
