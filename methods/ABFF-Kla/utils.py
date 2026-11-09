from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
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


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_device(device: str) -> str:
    return str(device or "gpu").lower()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}
    out["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    out["prauc"] = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    out["acc"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def search_best_threshold(y_true, y_prob) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_threshold)
    best_f1 = float(best_metrics["f1"])
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        if float(metrics["f1"]) > best_f1:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = float(metrics["f1"])
    return best_threshold, best_metrics


def to_jsonable(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def summarize_binary_split(name: str, y: np.ndarray, **extra) -> dict:
    y = np.asarray(y).astype(int)
    out = {
        "name": str(name),
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
    }
    out.update({k: to_jsonable(v) for k, v in extra.items()})
    return out


def compute_confusion(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def build_standard_metrics_payload(
    *,
    method: str,
    y_true,
    y_prob,
    raw_metrics: dict,
    threshold: float,
    primary_result: str,
    monitor: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict:
    confusion = compute_confusion(y_true, y_prob, threshold=threshold)
    specificity = float(confusion["tn"] / (confusion["tn"] + confusion["fp"])) if (confusion["tn"] + confusion["fp"]) else 0.0
    return {
        "method": method,
        "n": int(len(y_true)),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(raw_metrics["auc"]),
            "auprc": float(raw_metrics["prauc"]),
            "accuracy": float(raw_metrics["acc"]),
            "precision": float(raw_metrics["precision"]),
            "recall": float(raw_metrics["recall"]),
            "specificity": specificity,
            "f1": float(raw_metrics["f1"]),
            "mcc": float(raw_metrics["mcc"]),
        },
        "confusion_matrix": confusion,
        "selected_by": {"monitor": monitor, "best_epoch": None if best_epoch is None else int(best_epoch)},
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def build_standard_prediction_df(
    df_raw: pd.DataFrame,
    y_prob,
    threshold: float,
    sequence_col: str,
) -> pd.DataFrame:
    df_out = df_raw.copy().reset_index(drop=True)
    y_prob = np.asarray(y_prob).reshape(-1)
    standard = pd.DataFrame(
        {
            "RowId": np.arange(len(df_out), dtype=int),
            "Sequence": df_out[sequence_col].astype(str),
            "Prob": y_prob.astype(float),
            "PredLabel": (y_prob >= float(threshold)).astype(int),
        }
    )
    if "Label" in df_out.columns:
        standard.insert(2, "Label", df_out["Label"].astype(int))
    extra_cols = [column for column in df_out.columns if column not in {"Sequence", "Label"}]
    return pd.concat([standard, df_out[extra_cols]], axis=1)


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
        self._log_f = None
        self._old_stdout = None
        self._old_stderr = None

    def start(self) -> None:
        log_path = self._out_dir / "run.log"
        t0 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        self._log_f.write(f"===== run.log started at {t0} =====\n")
        self._log_f.flush()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeTextStream(self._old_stdout, self._log_f)
        sys.stderr = _TeeTextStream(self._old_stderr, self._log_f)

    def stop(self) -> None:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
            self._old_stdout = None
            self._old_stderr = None
        if self._log_f is not None:
            t1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self._log_f.write(f"===== run.log ended at {t1} =====\n")
            except Exception:
                pass
            self._log_f.close()
            self._log_f = None


@dataclass
class RunPaths:
    output_dir: Path
    model_dir: Path
    model_extra_dir: Path
    checkpoint_dir: Path
    results_dir: Path
    cache_dir: Path
    derived_contact_dir: Path
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    best_model_path: Path
    final_model_path: Path
    acid_pretrain_path: Path
    contact_pretrain_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    run_meta_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    model_dir = output_dir / "model"
    model_extra_dir = model_dir / "extra"
    checkpoint_dir = model_dir / "checkpoint"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    derived_contact_dir = cache_dir / "derived_contact"
    return RunPaths(
        output_dir=output_dir,
        model_dir=model_dir,
        model_extra_dir=model_extra_dir,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        cache_dir=cache_dir,
        derived_contact_dir=derived_contact_dir,
        run_config_path=output_dir / "run_config.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        preprocess_path=output_dir / "preprocess.json",
        training_log_path=output_dir / "training_log.csv",
        best_model_path=model_dir / "best_model.keras",
        final_model_path=model_dir / "final_model.keras",
        acid_pretrain_path=model_extra_dir / "acid0.h5",
        contact_pretrain_path=model_extra_dir / "contmap0.h5",
        val_metrics_path=results_dir / "val_metrics.json",
        test_metrics_path=results_dir / "test_metrics.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        run_meta_path=model_extra_dir / "run_meta.json",
    )
