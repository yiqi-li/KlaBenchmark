from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
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


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_name(train_tsv: str) -> str:
    train_parent = Path(train_tsv).resolve().parent.name
    return f"{train_parent}_{timestamp()}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    y_pred = (y_score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    balanced_accuracy = float((specificity + sensitivity) / 2.0)
    return {
        "auc_roc": auc,
        "auc_pr": ap,
        "auroc": auc,
        "auprc": ap,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "balanced_accuracy": balanced_accuracy,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def search_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_score, threshold=0.5)
    best_mcc = float(best_metrics["mcc"])
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = compute_metrics(y_true, y_score, threshold=float(threshold))
        if float(metrics["mcc"]) > best_mcc:
            best_threshold = float(threshold)
            best_mcc = float(metrics["mcc"])
            best_metrics = metrics
    return best_threshold, best_metrics


def split_stats(name: str, labels: np.ndarray) -> dict[str, int | str]:
    return {
        "name": str(name),
        "n": int(len(labels)),
        "pos": int(np.sum(np.asarray(labels) == 1)),
        "neg": int(np.sum(np.asarray(labels) == 0)),
    }


def build_standard_metrics_payload(
    *,
    method: str,
    raw_metrics: dict[str, float],
    primary_result: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict[str, object]:
    return {
        "method": method,
        "n": int(raw_metrics["tp"] + raw_metrics["tn"] + raw_metrics["fp"] + raw_metrics["fn"]),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(raw_metrics["auc_roc"]),
            "auprc": float(raw_metrics["auc_pr"]),
            "accuracy": float(raw_metrics["accuracy"]),
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
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    test_score: np.ndarray,
    threshold: float,
    esm2_prob: np.ndarray,
    lstm_prob: np.ndarray,
    feature_probabilities: dict[str, np.ndarray],
    feature_order: list[str],
) -> pd.DataFrame:
    result_df = source_df.copy().reset_index(drop=True)
    standard_df = pd.DataFrame(
        {
            "RowId": np.arange(len(result_df), dtype=int),
            "Sequence": result_df["Sequence"].astype(str),
            "Label": result_df["Label"].astype(int) if "Label" in result_df.columns else None,
            "Prob": test_score.astype(float),
            "PredLabel": (test_score >= float(threshold)).astype(int),
        }
    )
    if "Label" not in result_df.columns:
        standard_df = standard_df.drop(columns=["Label"])
    extra_cols = [column for column in result_df.columns if column not in {"Sequence", "Label"}]
    result_df = pd.concat([standard_df, result_df[extra_cols].reset_index(drop=True)], axis=1)
    result_df["prob_esm2"] = esm2_prob.astype(float)
    result_df["prob_lstm"] = lstm_prob.astype(float)
    for feature_name in feature_order:
        result_df[f"prob_{feature_name.lower()}"] = feature_probabilities[feature_name].astype(float)
    return result_df


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, sep="\t", index=False)


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
    def __init__(self, out_dir: Path):
        self._out_dir = Path(out_dir)
        self._log_file = None
        self._old_stdout = None
        self._old_stderr = None

    def start(self) -> None:
        log_path = self._out_dir / "run.log"
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self._log_file.write(f"===== run.log ended at {ended_at} =====\n")
            except Exception:
                pass
            self._log_file.close()
            self._log_file = None


@dataclass
class RunPaths:
    output_dir: Path
    model_dir: Path
    model_extra_dir: Path
    results_dir: Path
    cache_dir: Path
    feature_cache_dir: Path
    feature_model_dir: Path
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    checkpoint_path: Path
    gps_encoder_path: Path
    lstm_model_path: Path
    lstm_vocab_path: Path
    meta_feature_order_path: Path
    esm2_finetuned_dir: Path
    esm2_state_dict_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    run_meta_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    model_dir = output_dir / "model"
    model_extra_dir = model_dir / "extra"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    feature_model_dir = model_extra_dir / "feature_dnns"
    return RunPaths(
        output_dir=output_dir,
        model_dir=model_dir,
        model_extra_dir=model_extra_dir,
        results_dir=results_dir,
        cache_dir=cache_dir,
        feature_cache_dir=cache_dir / "features",
        feature_model_dir=feature_model_dir,
        run_config_path=output_dir / "run_config.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        preprocess_path=output_dir / "preprocess.json",
        training_log_path=output_dir / "training_log.csv",
        checkpoint_path=model_dir / "best_model.pth",
        gps_encoder_path=model_extra_dir / "gps_encoder.pt",
        lstm_model_path=model_extra_dir / "lstm.pth",
        lstm_vocab_path=model_extra_dir / "lstm_vocab.json",
        meta_feature_order_path=model_extra_dir / "meta_feature_order.json",
        esm2_finetuned_dir=model_extra_dir / "esm2_finetuned",
        esm2_state_dict_path=model_extra_dir / "esm2_state_dict.pth",
        val_metrics_path=results_dir / "val_metrics.json",
        test_metrics_path=results_dir / "test_metrics.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        run_meta_path=model_extra_dir / "run_meta.json",
    )
