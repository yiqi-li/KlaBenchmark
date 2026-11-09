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
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def format_float(value: float) -> str:
    try:
        if np.isnan(value):
            return "nan"
    except Exception:
        pass
    return f"{value:.4f}"


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
    results_dir: Path
    model_dir: Path
    checkpoint_dir: Path
    model_extra_dir: Path
    cache_dir: Path
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    test_metrics_path: Path
    val_metrics_path: Path
    curves_path: Path
    test_predictions_path: Path
    best_model_weights_path: Path
    input_encoder_path: Path
    output_spec_path: Path
    run_meta_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    results_dir = output_dir / "results"
    model_dir = output_dir / "model"
    checkpoint_dir = model_dir / "checkpoint"
    model_extra_dir = model_dir / "extra"
    cache_dir = output_dir / "cache"
    return RunPaths(
        output_dir=output_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        model_extra_dir=model_extra_dir,
        cache_dir=cache_dir,
        run_config_path=output_dir / "run_config.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        preprocess_path=output_dir / "preprocess.json",
        training_log_path=output_dir / "training_log.csv",
        test_metrics_path=results_dir / "test_metrics.json",
        val_metrics_path=results_dir / "val_metrics.json",
        curves_path=results_dir / "curves.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        best_model_weights_path=model_dir / "best_model.weights.h5",
        input_encoder_path=model_extra_dir / "input_encoder.pkl",
        output_spec_path=model_extra_dir / "output_spec.pkl",
        run_meta_path=model_extra_dir / "run_meta.json",
    )


def split_stats(name: str, df: pd.DataFrame) -> dict[str, int | str]:
    pos = int(df["Label"].sum())
    n = int(len(df))
    neg = n - pos
    return {"name": str(name), "n": n, "pos": pos, "neg": neg}


def build_standard_metrics_payload(
    *,
    method: str,
    raw_metrics: dict,
    primary_result: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict[str, Any]:
    confusion_matrix = raw_metrics["confusion_matrix"]
    precision = float(confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"])) if (
        confusion_matrix["tp"] + confusion_matrix["fp"]
    ) else 0.0
    recall = float(raw_metrics["Sensitivity"])
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "method": method,
        "n": int(raw_metrics["n"]),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(raw_metrics["AUROC"]),
            "auprc": float(raw_metrics["AUPRC"]),
            "accuracy": float(raw_metrics["Accuracy"]),
            "precision": precision,
            "recall": recall,
            "specificity": float(raw_metrics["Specificity"]),
            "f1": f1,
            "mcc": float(raw_metrics["MCC"]),
        },
        "confusion_matrix": {key: int(value) for key, value in confusion_matrix.items()},
        "selected_by": {
            "monitor": "val_auroc",
            "best_epoch": None if best_epoch is None else int(best_epoch),
        },
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    def _threshold_metrics(y_true_: np.ndarray, y_score_: np.ndarray, decision_threshold: float) -> dict:
        y_pred_ = (y_score_ >= float(decision_threshold)).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true_, y_pred_, labels=[0, 1]).ravel()
        acc_ = float(accuracy_score(y_true_, y_pred_))
        sn_ = float(tp_ / (tp_ + fn_)) if (tp_ + fn_) > 0 else float("nan")
        sp_ = float(tn_ / (tn_ + fp_)) if (tn_ + fp_) > 0 else float("nan")
        mcc_ = float(matthews_corrcoef(y_true_, y_pred_))
        return {
            "Accuracy": acc_,
            "Sensitivity": sn_,
            "Specificity": sp_,
            "MCC": mcc_,
            "confusion_matrix": {"tn": int(tn_), "fp": int(fp_), "fn": int(fn_), "tp": int(tp_)},
        }

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = float(auc(recall, precision))

    threshold_metrics = _threshold_metrics(y_true, y_score, threshold)
    return {
        "n": int(len(y_true)),
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": float(threshold_metrics["Accuracy"]),
        "Sensitivity": float(threshold_metrics["Sensitivity"]),
        "Specificity": float(threshold_metrics["Specificity"]),
        "MCC": float(threshold_metrics["MCC"]),
        "confusion_matrix": threshold_metrics["confusion_matrix"],
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def save_curves(metrics: dict, path: Path) -> None:
    curves_payload = {
        "roc": {
            "fpr": metrics["roc_curve"]["fpr"],
            "tpr": metrics["roc_curve"]["tpr"],
            "auroc": metrics["AUROC"],
        },
        "pr": {
            "precision": metrics["pr_curve"]["precision"],
            "recall": metrics["pr_curve"]["recall"],
            "auprc": metrics["AUPRC"],
        },
    }
    save_json(path, curves_payload)


def build_prediction_dataframe(raw_df: pd.DataFrame, y_score: np.ndarray, threshold: float) -> pd.DataFrame:
    prediction_df = raw_df.copy().reset_index(drop=True)
    prediction_df.insert(0, "RowId", np.arange(len(prediction_df), dtype=int))
    prediction_df["Prob"] = y_score.astype(float)
    standard = {
        "RowId": np.arange(len(prediction_df), dtype=int),
        "Sequence": prediction_df["Sequence"].astype(str),
        "Prob": prediction_df["Prob"].astype(float),
        "PredLabel": (prediction_df["Prob"] >= float(threshold)).astype(int),
    }
    if "Label" in prediction_df.columns:
        standard["Label"] = prediction_df["Label"].astype(int)

    preferred_order = ["RowId", "Sequence"]
    if "Label" in standard:
        preferred_order.append("Label")
    preferred_order.extend(["Prob", "PredLabel"])
    standard_df = pd.DataFrame(standard)[preferred_order]

    extra_cols = [column for column in raw_df.columns if column not in {"Sequence", "Label"}]
    return pd.concat([standard_df, raw_df[extra_cols].reset_index(drop=True)], axis=1)


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, sep="\t", index=False)
