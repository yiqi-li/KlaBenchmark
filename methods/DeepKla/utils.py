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
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_name(train_tsv: str) -> str:
    parent = Path(train_tsv).resolve().parent.name or "data"
    return f"{parent}_{timestamp()}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
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
    import warnings

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0

    output: dict[str, float] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float("nan"),
        "sn": float(sn),
        "sp": float(sp),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    try:
        output["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        output["auc"] = float("nan")
    try:
        output["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        output["auprc"] = float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            output["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            output["mcc"] = float("nan")
    return output


def search_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    best_mcc = float(best_metrics["mcc"]) if not np.isnan(best_metrics["mcc"]) else -float("inf")
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        mcc = float(metrics["mcc"]) if not np.isnan(metrics["mcc"]) else -float("inf")
        if mcc > best_mcc:
            best_threshold = float(threshold)
            best_mcc = mcc
            best_metrics = metrics
    return best_threshold, best_metrics


def split_stats(name: str, y: np.ndarray) -> dict[str, int | str]:
    return {
        "name": str(name),
        "n": int(len(y)),
        "pos": int(np.sum(np.asarray(y) == 1)),
        "neg": int(np.sum(np.asarray(y) == 0)),
    }


def build_standard_metrics_payload(
    *,
    method: str,
    n: int,
    metrics_dict: dict[str, float],
    primary_result: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict[str, object]:
    return {
        "method": method,
        "n": int(n),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(metrics_dict["auc"]),
            "auprc": float(metrics_dict["auprc"]),
            "accuracy": float(metrics_dict["acc"]),
            "precision": float(metrics_dict["precision"]),
            "recall": float(metrics_dict["sn"]),
            "specificity": float(metrics_dict["sp"]),
            "f1": float(metrics_dict["f1"]),
            "mcc": float(metrics_dict["mcc"]),
        },
        "confusion_matrix": {
            "tn": int(metrics_dict["tn"]),
            "fp": int(metrics_dict["fp"]),
            "fn": int(metrics_dict["fn"]),
            "tp": int(metrics_dict["tp"]),
        },
        "selected_by": {"monitor": "val_auroc", "best_epoch": None if best_epoch is None else int(best_epoch)},
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def build_prediction_dataframe(rows: list[dict[str, str]], extra_cols: list[str], y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "RowId": np.arange(len(rows), dtype=int),
            "Sequence": [str(row.get("Sequence", "")) for row in rows],
            "Label": [int(row.get("Label", 0)) for row in rows] if rows and "Label" in rows[0] else None,
            "Prob": y_prob.astype(float),
            "PredLabel": (y_prob >= float(threshold)).astype(int),
        }
    )
    if rows and "Label" not in rows[0]:
        pred_df = pred_df.drop(columns=["Label"])
    if extra_cols:
        pred_df = pd.concat([pred_df, pd.DataFrame(rows)[extra_cols].reset_index(drop=True)], axis=1)
    return pred_df


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
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    best_model_path: Path
    final_model_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    run_meta_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    model_dir = output_dir / "model"
    model_extra_dir = model_dir / "extra"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    return RunPaths(
        output_dir=output_dir,
        model_dir=model_dir,
        model_extra_dir=model_extra_dir,
        results_dir=results_dir,
        cache_dir=cache_dir,
        run_config_path=output_dir / "run_config.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        preprocess_path=output_dir / "preprocess.json",
        training_log_path=output_dir / "training_log.csv",
        best_model_path=model_dir / "best_model.keras",
        final_model_path=model_dir / "final_model.keras",
        val_metrics_path=results_dir / "val_metrics.json",
        test_metrics_path=results_dir / "test_metrics.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        run_meta_path=model_extra_dir / "run_meta.json",
    )
