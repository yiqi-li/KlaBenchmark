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
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def default_run_name(train_tsv: str) -> str:
    parent = Path(train_tsv).resolve().parent.name or "data"
    return f"{parent}_{timestamp()}"


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


@dataclass
class Metrics:
    acc: float
    f1: float
    mcc: float
    auroc: float
    auprc: float
    sn: float
    sp: float


def split_stats(name: str, y: np.ndarray) -> dict[str, int | str]:
    return {
        "name": str(name),
        "n": int(len(y)),
        "pos": int(np.sum(np.asarray(y) == 1)),
        "neg": int(np.sum(np.asarray(y) == 0)),
    }


def compute_confusion(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob_pos).reshape(-1) >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def compute_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_prob_pos >= float(threshold)).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    auroc = float("nan")
    auprc = float("nan")
    try:
        auroc = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        pass
    try:
        auprc = float(average_precision_score(y_true, y_prob_pos))
    except Exception:
        pass

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return Metrics(acc=acc, f1=f1, mcc=mcc, auroc=auroc, auprc=auprc, sn=sn, sp=sp)


def search_best_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray) -> tuple[float, Metrics]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_prob_pos, threshold=0.5)
    best_mcc = float(best_metrics.mcc)
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = compute_metrics(y_true, y_prob_pos, threshold=float(threshold))
        if float(metrics.mcc) > best_mcc:
            best_mcc = float(metrics.mcc)
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def build_standard_metrics_payload(
    *,
    method: str,
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    metrics_obj: Metrics,
    threshold: float,
    primary_result: str,
    best_epoch,
    predictions_relpath: str,
    best_model_relpath: str,
) -> dict[str, object]:
    confusion = compute_confusion(y_true, y_prob_pos, threshold=float(threshold))
    return {
        "method": method,
        "n": int(len(y_true)),
        "primary_result": primary_result,
        "metrics": {
            "auroc": float(metrics_obj.auroc),
            "auprc": float(metrics_obj.auprc),
            "accuracy": float(metrics_obj.acc),
            "precision": float(confusion["tp"] / (confusion["tp"] + confusion["fp"])) if (confusion["tp"] + confusion["fp"]) else 0.0,
            "recall": float(metrics_obj.sn),
            "specificity": float(metrics_obj.sp),
            "f1": float(metrics_obj.f1),
            "mcc": float(metrics_obj.mcc),
        },
        "confusion_matrix": confusion,
        "selected_by": {"monitor": "val_auroc", "best_epoch": None if best_epoch is None else int(best_epoch)},
        "paths": {
            "predictions": predictions_relpath,
            "best_model": best_model_relpath,
        },
    }


def build_prediction_dataframe(source_df: pd.DataFrame, y_prob_pos: np.ndarray, threshold: float) -> pd.DataFrame:
    standard = pd.DataFrame(
        {
            "RowId": np.arange(len(source_df), dtype=int),
            "Sequence": source_df["Sequence"].astype(str),
            "Prob": y_prob_pos.astype(np.float64),
            "PredLabel": (y_prob_pos >= float(threshold)).astype(int),
        }
    )
    if "Label" in source_df.columns:
        standard.insert(2, "Label", source_df["Label"].astype(int))
    extra_cols = [column for column in source_df.columns if column not in {"Sequence", "Label"}]
    return pd.concat([standard, source_df[extra_cols].reset_index(drop=True)], axis=1)


def run_predict_proba_with_index(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_idx = []
    all_true = []
    all_prob = []
    with torch.no_grad():
        for idx, input_ids, desc_feats, labels in loader:
            input_ids = input_ids.to(device)
            desc_feats = desc_feats.to(device)
            logits = model(input_ids, desc_feats)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_idx.append(idx.detach().cpu().numpy())
            all_true.append(labels.detach().cpu().numpy())
            all_prob.append(prob)

    idx = np.concatenate(all_idx, axis=0) if all_idx else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob, axis=0) if all_prob else np.array([], dtype=np.float64)

    num_samples = int(idx.shape[0])
    assert y_true.shape[0] == num_samples and y_prob.shape[0] == num_samples, (
        f"Prediction alignment broken: lens idx={num_samples} true={y_true.shape[0]} prob={y_prob.shape[0]}"
    )
    if num_samples > 0:
        idx_min = int(idx.min())
        idx_max = int(idx.max())
        assert idx_min == 0 and idx_max == num_samples - 1, f"Index range invalid: min={idx_min} max={idx_max} n={num_samples}"
        uniq = np.unique(idx)
        assert uniq.shape[0] == num_samples, f"Duplicate indices found: unique={uniq.shape[0]} n={num_samples}"
        order = np.argsort(idx)
        idx = idx[order]
        y_true = y_true[order]
        y_prob = y_prob[order]
        assert np.all(idx == np.arange(num_samples, dtype=idx.dtype)), "Index is not a permutation of 0..N-1 after sort"

    return idx, y_true, y_prob


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
    model_dir: Path
    model_extra_dir: Path
    results_dir: Path
    cache_dir: Path
    run_config_path: Path
    dataset_stats_path: Path
    preprocess_path: Path
    training_log_path: Path
    checkpoint_path: Path
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
        checkpoint_path=model_dir / "best_model.pt",
        val_metrics_path=results_dir / "val_metrics.json",
        test_metrics_path=results_dir / "test_metrics.json",
        test_predictions_path=results_dir / "test_predictions.tsv",
        run_meta_path=model_extra_dir / "run_meta.json",
    )
