from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

METHOD_NAME = "HybridKla"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "HybridKla" / "KlaAtlas_benchmark_summary" / "protein_split_dataset"
DEFAULT_TRAIN_TSV = str(DEFAULT_DATA_DIR / "train.tsv")
DEFAULT_VAL_TSV = str(DEFAULT_DATA_DIR / "val.tsv")
DEFAULT_TEST_TSV = str(DEFAULT_DATA_DIR / "test.tsv")
DEFAULT_ESM2_DIR = "facebook/esm2_t30_150M_UR50D"
DEFAULT_OUT_ROOT = str(Path(__file__).resolve().parent / "runs")


@dataclass
class Config:
    train_tsv: str = DEFAULT_TRAIN_TSV
    val_tsv: str = DEFAULT_VAL_TSV
    test_tsv: str = DEFAULT_TEST_TSV
    esm2_dir: str = DEFAULT_ESM2_DIR
    output_dir: str = ""
    out_root: str = DEFAULT_OUT_ROOT
    run_name: str = ""
    checkpoint_path: str = ""
    run_meta_path: str = ""
    gps_encoder_path: str = ""
    lstm_model_path: str = ""
    lstm_vocab_path: str = ""
    meta_feature_order_path: str = ""
    feature_model_dir: str = ""
    esm2_finetuned_dir: str = ""
    esm2_state_dict_path: str = ""
    seed: int = 42
    device: str = "cuda"
    gpu: str = ""
    seq_len: int = 51
    batch_size: int = 512
    epochs: int = 100
    feature_lr: float = 1.5018e-3
    lstm_lr: float = 3.1021e-4
    esm2_lr: float = 1.3638e-4
    meta_lr: float = 1.0701e-5
    esm2_weight_decay: float = 0.05
    esm2_warmup_steps: int = 150
    enable_acf: bool = True
    enable_aaindex: bool = True
    enable_obc: bool = True
    enable_gps: bool = True
    enable_cksaap: bool = True
    enable_pseaac: bool = True
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
    parser = argparse.ArgumentParser(description="Train and evaluate HybridKla.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--train_tsv", default=None)
    parser.add_argument("--val_tsv", default=None)
    parser.add_argument("--test_tsv", default=None)
    parser.add_argument("--esm2_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--feature_lr", type=float, default=None)
    parser.add_argument("--lstm_lr", type=float, default=None)
    parser.add_argument("--esm2_lr", type=float, default=None)
    parser.add_argument("--meta_lr", type=float, default=None)
    parser.add_argument("--esm2_weight_decay", type=float, default=None)
    parser.add_argument("--esm2_warmup_steps", type=int, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    parser.add_argument("--enable_acf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_aaindex", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_obc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_gps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_cksaap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_pseaac", action=argparse.BooleanOptionalAction, default=None)
    return parser


def make_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained HybridKla checkpoint and run prediction.")
    parser.add_argument("--config", default="", help="Optional JSON config file.")

    parser.add_argument("--checkpoint_path", default=None, help="Path to the saved meta-model checkpoint.")
    parser.add_argument("--test_tsv", "--input_tsv", dest="test_tsv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_meta_path", default=None)
    parser.add_argument("--gps_encoder_path", default=None)
    parser.add_argument("--lstm_model_path", default=None)
    parser.add_argument("--lstm_vocab_path", default=None)
    parser.add_argument("--meta_feature_order_path", default=None)
    parser.add_argument("--feature_model_dir", default=None)
    parser.add_argument("--esm2_finetuned_dir", default=None)
    parser.add_argument("--esm2_state_dict_path", default=None)
    parser.add_argument("--esm2_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", default=None)
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
