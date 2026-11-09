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
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score



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


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def save_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def default_run_name(train_tsv: str) -> str:
    dataset_name = Path(train_tsv).parent.name or "run"
    return f"{dataset_name}_{timestamp()}"


def summarize_split(name: str, y: np.ndarray) -> str:
    n = int(y.shape[0])
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    pos_rate = (pos / n) if n else 0.0
    return f"{name}: n={n} pos={pos} neg={neg} pos_rate={pos_rate:.4f}"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    pre = tp / max(1, tp + fp)
    sn = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if (tp + tn + fp + fn) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")

    return {
        "acc": float(acc),
        "pre": float(pre),
        "sn": float(sn),
        "sp": float(sp),
        "mcc": float(mcc),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def search_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    best_mcc = float(best_metrics["mcc"])
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        if float(metrics["mcc"]) > best_mcc:
            best_threshold = float(threshold)
            best_mcc = float(metrics["mcc"])
            best_metrics = metrics
    return best_threshold, best_metrics


def split_stats(name: str, y: np.ndarray) -> dict[str, int]:
    return {
        "name": str(name),
        "n": int(y.shape[0]),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
    }


def build_standard_metrics_payload(
    *,
    method: str,
    n: int,
    raw_metrics: dict[str, float],
    primary_result: str,
    monitor: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict[str, object]:
    precision = float(raw_metrics["pre"])
    recall = float(raw_metrics["sn"])
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "method": method,
        "n": int(n),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(raw_metrics["auroc"]),
            "auprc": float(raw_metrics["auprc"]),
            "accuracy": float(raw_metrics["acc"]),
            "precision": precision,
            "recall": recall,
            "specificity": float(raw_metrics["sp"]),
            "f1": float(f1),
            "mcc": float(raw_metrics["mcc"]),
        },
        "confusion_matrix": {
            "tn": int(raw_metrics["tn"]),
            "fp": int(raw_metrics["fp"]),
            "fn": int(raw_metrics["fn"]),
            "tp": int(raw_metrics["tp"]),
        },
        "selected_by": {"monitor": monitor, "best_epoch": None if best_epoch is None else int(best_epoch)},
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def build_prediction_df(test_df: pd.DataFrame, y_prob: np.ndarray, cutoff: float) -> pd.DataFrame:
    y_prob = np.asarray(y_prob).astype(float)
    if int(test_df.shape[0]) != int(y_prob.shape[0]):
        raise RuntimeError(f"Prediction count mismatch: pred={y_prob.shape[0]} test_rows={test_df.shape[0]}")
    out = pd.DataFrame(
        {
            "RowId": np.arange(int(test_df.shape[0]), dtype=int),
            "Sequence": test_df["Sequence"].astype(str),
            "Prob": y_prob,
            "PredLabel": (y_prob >= float(cutoff)).astype(int),
        }
    )
    if "Label" in test_df.columns:
        out.insert(2, "Label", test_df["Label"].astype(int))
    extra_cols = [column for column in test_df.columns if column not in {"Sequence", "Label"}]
    return pd.concat([out, test_df[extra_cols].reset_index(drop=True)], axis=1)


def save_predictions(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, sep="\t", index=False)


class _TeeTextStream:
    __slots__ = ("_streams",)

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass
        try:
            return len(data)
        except Exception:
            return 0

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
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
    checkpoint_dir: Path
    model_extra_dir: Path
    results_dir: Path
    cache_dir: Path
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    best_model_path: Path
    top3_meta_path: Path
    best_meta_path: Path
    best_epoch_state_dict_path: Path
    val_metrics_path: Path
    test_metrics_path: Path
    test_predictions_path: Path
    run_meta_path: Path


def build_run_paths(output_dir: Path) -> RunPaths:
    model_dir = output_dir / "model"
    checkpoint_dir = model_dir / "checkpoint"
    model_extra_dir = model_dir / "extra"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    return RunPaths(
        output_dir=output_dir,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        model_extra_dir=model_extra_dir,
        results_dir=results_dir,
        cache_dir=cache_dir,
        run_config_path=output_dir / "run_config.json",
        dataset_stats_path=output_dir / "dataset_stats.json",
        preprocess_path=output_dir / "preprocess.json",
        training_log_path=output_dir / "training_log.csv",
        best_model_path=model_dir / "best_model.pt",
        top3_meta_path=checkpoint_dir / "top3_meta.json",
        best_meta_path=model_extra_dir / "best_meta.json",
        best_epoch_state_dict_path=model_extra_dir / "best_epoch_state_dict.pt",
        val_metrics_path=results_dir / "val_metrics.json",
        test_metrics_path=results_dir / "test_metrics.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        run_meta_path=model_extra_dir / "run_meta.json",
    )
