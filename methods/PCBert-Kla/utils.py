from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)



def prepare_input_sequence(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8").strip().upper()
    return str(raw).strip().upper()


def raw_sequence_length(raw) -> int:
    return len(prepare_input_sequence(raw))


def center_align_length(sequence: str, target_len: int, pad_char: str = "_") -> str:
    if target_len <= 0:
        return sequence
    n = len(sequence)
    if n == target_len:
        return sequence
    if n > target_len:
        excess = n - target_len
        left_cut = excess // 2
        return sequence[left_cut : left_cut + target_len]
    pad_total = target_len - n
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    return (pad_char * left_pad) + sequence + (pad_char * right_pad)


def log_seq_len_adjustments(
    lengths_before: Iterable[int],
    target_len: int,
    *,
    method_label: str,
    split_label: str = "",
) -> None:
    lengths = list(lengths_before)
    if not lengths:
        return
    n_crop = sum(1 for length in lengths if length > target_len)
    n_pad = sum(1 for length in lengths if length < target_len)
    if n_crop == 0 and n_pad == 0:
        return
    split_part = f" {split_label}" if split_label else ""
    print(
        f"[{method_label}]{split_part} seq_len={target_len}: "
        f"{n_crop} sequence(s) center-cropped, {n_pad} sequence(s) center-padded "
        f"(raw length includes '_'; special tokens such as [CLS] are not counted)"
    )


def align_raw_sequence(
    raw,
    target_len: int,
    *,
    pad_char: str = "_",
    normalize_chars: Optional[Callable[[str], str]] = None,
) -> Tuple[str, int]:
    sequence = prepare_input_sequence(raw)
    length_before = len(sequence)
    if normalize_chars is not None:
        sequence = normalize_chars(sequence)
    aligned = center_align_length(sequence, int(target_len), pad_char=pad_char)
    return aligned, length_before


def align_sequence_batch(
    seqs: Iterable,
    target_len: int,
    *,
    method_label: str,
    split_label: str = "",
    pad_char: str = "_",
    normalize_chars: Optional[Callable[[str], str]] = None,
    warn: bool = True,
) -> List[str]:
    lengths_before: List[int] = []
    aligned: List[str] = []
    for raw in seqs:
        sequence, length_before = align_raw_sequence(
            raw,
            target_len,
            pad_char=pad_char,
            normalize_chars=normalize_chars,
        )
        lengths_before.append(length_before)
        aligned.append(sequence)
    if warn:
        log_seq_len_adjustments(
            lengths_before,
            int(target_len),
            method_label=method_label,
            split_label=split_label,
        )
    return aligned


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_run_name(train_tsv: str) -> str:
    train_parent = Path(train_tsv).resolve().parent.name
    return f"{train_parent}_{timestamp()}"


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _TeeTextStream:
    __slots__ = ("_streams",)

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            try:
                stream.flush()
            except Exception:
                pass
        try:
            return len(data)
        except Exception:
            return 0

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return self._streams[0].isatty() if self._streams else False

    def fileno(self):
        return self._streams[0].fileno()

    def writable(self):
        return True


class TeeRunLog:
    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._log_file = None
        self._old_stdout = None
        self._old_stderr = None

    def start(self) -> None:
        log_path = self._output_dir / "run.log"
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._log_file.write(f"===== run.log started at {started_at} =====\n")
        self._log_file.flush()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeTextStream(self._old_stdout, self._log_file)
        sys.stderr = _TeeTextStream(self._old_stderr, self._log_file)

    def stop(self) -> None:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
            self._old_stdout = None
            self._old_stderr = None
        if self._log_file is not None:
            ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                self._log_file.write(f"===== run.log ended at {ended_at} =====\n")
            except Exception:
                pass
            self._log_file.close()
            self._log_file = None


@dataclass
class RunPaths:
    output_dir: Path
    cache_dir: Path
    results_dir: Path
    model_dir: Path
    model_extra_dir: Path
    scaler_path: Path
    model_path: Path
    metrics_path: Path
    run_meta_path: Path
    run_config_path: Path
    preprocess_path: Path
    dataset_stats_path: Path
    training_log_path: Path
    test_predictions_path: Path
    val_metrics_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    model_dir = output_dir / "model"
    model_extra_dir = model_dir / "extra"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    return RunPaths(
        output_dir=output_dir,
        cache_dir=cache_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        model_extra_dir=model_extra_dir,
        scaler_path=model_extra_dir / "scaler_model.pkl",
        model_path=model_dir / "best_model.pt",
        metrics_path=results_dir / "test_metrics.json",
        run_meta_path=model_extra_dir / "run_meta.json",
        run_config_path=output_dir / "run_config.json",
        preprocess_path=output_dir / "preprocess.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        training_log_path=output_dir / "training_log.csv",
        test_predictions_path=results_dir / "test_predictions.tsv",
        val_metrics_path=results_dir / "val_metrics.json",
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")

    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": specificity,
        "auc": auc,
        "auprc": auprc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def split_stats(name: str, labels: np.ndarray) -> dict[str, int | str]:
    array = np.asarray(labels)
    return {
        "name": str(name),
        "n": int(len(array)),
        "pos": int(np.sum(array == 1)),
        "neg": int(np.sum(array == 0)),
    }


def build_standard_metrics_payload(
    *,
    method: str,
    raw_metrics: dict[str, float | int],
    primary_result: str,
    best_epoch,
    predictions_relpath: str | None,
    best_model_relpath: str,
) -> dict[str, Any]:
    paths = {"best_model": best_model_relpath}
    if predictions_relpath:
        paths["predictions"] = predictions_relpath
    return {
        "method": method,
        "n": int(raw_metrics["n"]),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(raw_metrics["auc"]),
            "auprc": float(raw_metrics["auprc"]),
            "accuracy": float(raw_metrics["acc"]),
            "precision": float(raw_metrics["precision"]),
            "recall": float(raw_metrics["recall"]),
            "specificity": float(raw_metrics["specificity"]),
            "f1": float(raw_metrics["f1"]),
            "mcc": float(raw_metrics["mcc"]),
        },
        "confusion_matrix": {
            "tn": int(raw_metrics["tn"]),
            "fp": int(raw_metrics["fp"]),
            "fn": int(raw_metrics["fn"]),
            "tp": int(raw_metrics["tp"]),
        },
        "selected_by": {
            "monitor": "val_auroc",
            "best_epoch": None if best_epoch is None else int(best_epoch),
        },
        "paths": paths,
    }


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    seq_col: str,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    frame = source_df.copy()
    frame.columns = [column.strip() for column in frame.columns]

    standard = {
        "RowId": np.arange(len(frame), dtype=int),
        "Sequence": frame[seq_col].astype(str),
        "Prob": probabilities.astype(float),
        "PredLabel": (probabilities >= float(threshold)).astype(int),
    }
    if "Label" in frame.columns:
        standard["Label"] = frame["Label"].astype(int)

    preferred_order = ["RowId", "Sequence"]
    if "Label" in standard:
        preferred_order.append("Label")
    preferred_order.extend(["Prob", "PredLabel"])

    standard_df = pd.DataFrame(standard)[preferred_order]
    extra_cols = [column for column in frame.columns if column not in {seq_col, "Label"}]
    return pd.concat([standard_df, frame[extra_cols].reset_index(drop=True)], axis=1)


def validate_prediction_alignment(source_df: pd.DataFrame, seq_col: str, expected_sequences: np.ndarray) -> None:
    source_sequences = source_df[seq_col].astype(str).to_numpy()
    if len(source_sequences) != len(expected_sequences) or not np.array_equal(source_sequences, expected_sequences.astype(str)):
        raise RuntimeError("Prediction alignment check failed: source rows and extracted sequences do not match.")


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")
