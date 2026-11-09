from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import align_raw_sequence, log_seq_len_adjustments, raw_sequence_length


def read_tsv(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "Sequence" not in df.columns:
        raise ValueError(f"TSV must contain the Sequence column: {path}")
    if require_label and "Label" not in df.columns:
        raise ValueError(f"TSV must contain Sequence/Label columns: {path}, actual={list(df.columns)}")
    df = df.copy()
    df["Sequence"] = df["Sequence"].astype(str)
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)
    return df


def adjust_seq_len(seq: str, seq_len: int, pad_char: str = "_") -> str:
    aligned, _ = align_raw_sequence(seq, seq_len, pad_char=pad_char)
    return aligned


def clean_seq_for_models(seq: str, seq_len: int) -> str:
    sequence = adjust_seq_len(seq, seq_len=seq_len, pad_char="_")
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([char if (char == "_" or char == "X" or char in allowed) else "X" for char in sequence])


def clean_seq_for_features(seq: str, seq_len: int) -> str:
    return clean_seq_for_models(seq, seq_len=seq_len).replace("_", "X")


def clean_seq_for_esm2(seq: str, seq_len: int) -> str:
    sequence = clean_seq_for_models(seq, seq_len=seq_len).replace("_", "X")
    allowed = set("ACDEFGHIKLMNPQRSTVWYX")
    return "".join([char if char in allowed else "X" for char in sequence])


def prepare_sequences(
    df: pd.DataFrame,
    seq_len: int,
    *,
    method_label: str = "HybridKla",
    split_label: str = "",
    warn: bool = True,
) -> dict[str, list[str]]:
    raw_sequences = df["Sequence"].astype(str).tolist()
    if warn:
        log_seq_len_adjustments(
            [raw_sequence_length(seq) for seq in raw_sequences],
            int(seq_len),
            method_label=method_label,
            split_label=split_label,
        )
    return {
        "raw": raw_sequences,
        "model": [clean_seq_for_models(sequence, seq_len) for sequence in raw_sequences],
        "feature": [clean_seq_for_features(sequence, seq_len) for sequence in raw_sequences],
        "esm2": [clean_seq_for_esm2(sequence, seq_len) for sequence in raw_sequences],
    }


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_lstm_vocab() -> Dict[str, int]:
    trans = {"_": 0}
    for i, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1):
        trans[amino_acid] = i
    trans["X"] = 21
    trans["B"] = 21
    trans["U"] = 21
    trans["Z"] = 21
    trans["O"] = 21
    trans["J"] = 21
    return trans


def seqs_to_lstm_ids(seqs: List[str], trans: Dict[str, int]) -> np.ndarray:
    output = np.zeros((len(seqs), len(seqs[0])), dtype=np.int64)
    for i, sequence in enumerate(seqs):
        output[i] = np.array([trans.get(char, trans["X"]) for char in sequence], dtype=np.int64)
    return output


AA20 = list("ACDEFGHIKLMNPQRSTVWY")

TOP10_TSV = """name\tA\tL\tR\tK\tN\tM\tD\tF\tC\tP\tQ\tS\tE\tT\tG\tW\tH\tY\tI\tV
115\t-0.5\t-1.8\t3.0\t3.0\t0.2\t-1.3\t3.0\t-2.5\t-1.0\t0.0\t0.2\t0.3\t3.0\t-0.4\t0.0\t-3.4\t-0.5\t-2.3\t-1.8\t-1.5
153\t-0.5\t-1.8\t3.0\t3.0\t0.2\t-1.3\t2.5\t-2.5\t-1.0\t-1.4\t0.2\t0.3\t2.5\t-0.4\t0.0\t-3.4\t-0.5\t-2.3\t-1.8\t-1.5
525\t0.39\t1.82\t-3.95\t-2.77\t-1.91\t0.96\t-3.81\t2.27\t0.25\t0\t-1.30\t-1.24\t-2.91\t-1\t0\t2.13\t-0.64\t1.47\t1.82\t1.30
526\t0.18\t0.41\t-5.40\t-2.53\t-1.30\t0.44\t-2.36\t0.50\t0.27\t-0.20\t-1.22\t-0.40\t-2.10\t-0.34\t0.09\t-0.01\t-1.48\t-0.08\t0.37\t0.32
499\t-0.17\t-0.28\t0.37\t0.32\t0.18\t-0.26\t0.37\t-0.41\t-0.06\t0.13\t0.26\t0.05\t0.15\t0.02\t0.01\t-0.15\t-0.02\t-0.09\t-0.28\t-0.17
88\t0\t0\t1\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0
543\t-1.6\t-2.8\t12.3\t8.8\t4.8\t-3.4\t9.2\t-3.7\t-2.0\t0.2\t4.1\t-0.6\t8.2\t-1.2\t-1.0\t-1.9\t3.0\t0.7\t-3.1\t-2.6
68\t0.25\t0.53\t-1.76\t-1.10\t-0.64\t0.26\t-0.72\t0.61\t0.04\t-0.07\t-0.69\t-0.26\t-0.62\t-0.18\t0.16\t0.37\t-0.40\t0.02\t0.73\t0.54
252\t-6.70\t-11.70\t51.50\t36.80\t20.10\t-14.20\t38.50\t-15.50\t-8.40\t0.80\t17.20\t-2.50\t34.30\t-5\t-4.20\t-7.90\t12.60\t2.90\t-13\t-10.90
522\t-0.31\t-0.53\t1.30\t1.79\t0.49\t-0.38\t0.58\t-0.45\t-0.87\t0.34\t0.70\t0.10\t0.68\t0.21\t-0.33\t-0.27\t0.13\t0.40\t-0.66\t-0.62
"""


def load_top10() -> Tuple[List[str], Dict[str, List[float]]]:
    rows = TOP10_TSV.strip().splitlines()
    header = rows[0].split("\t")[1:]
    feats: Dict[str, List[float]] = {}
    for line in rows[1:]:
        parts = line.split("\t")
        feats[parts[0]] = [float(value) for value in parts[1:]]
    return header, feats


def aaindex_encode(seqs: List[str], aa_header: List[str], feat_values: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    aa_to_idx = {aa: i for i, aa in enumerate(aa_header)}
    feat_names = list(feat_values.keys())
    num_features = len(feat_names)
    seq_len = len(seqs[0])

    seq_idx = np.full((len(seqs), seq_len), -1, dtype=np.int64)
    for i, sequence in enumerate(seqs):
        for j, char in enumerate(sequence):
            seq_idx[i, j] = aa_to_idx.get(char, -1)

    lookup = np.zeros((num_features, len(aa_header) + 1), dtype=np.float32)
    for fi, name in enumerate(feat_names):
        values = feat_values[name]
        lookup[fi, : len(aa_header)] = np.array(values, dtype=np.float32)
        lookup[fi, len(aa_header)] = 0.0

    idx_safe = seq_idx.copy()
    idx_safe[idx_safe < 0] = len(aa_header)

    aaindex = np.zeros((len(seqs), num_features * seq_len), dtype=np.float32)
    for fi in range(num_features):
        values = lookup[fi][idx_safe]
        aaindex[:, fi * seq_len : (fi + 1) * seq_len] = values

    acf = np.zeros((len(seqs), num_features * seq_len), dtype=np.float32)
    for fi in range(num_features):
        values = aaindex[:, fi * seq_len : (fi + 1) * seq_len]
        for k in range(seq_len):
            if k == 0:
                acf[:, fi * seq_len + k] = np.round(np.mean(values * values, axis=1), 2)
            else:
                acf[:, fi * seq_len + k] = np.round(np.mean(values[:, : seq_len - k] * values[:, k:], axis=1), 2)
    return acf, aaindex


def obc_onehot_flat(seqs: List[str], alphabet: List[str]) -> np.ndarray:
    seq_len = len(seqs[0])
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    output = np.zeros((len(seqs), seq_len * len(alphabet)), dtype=np.float32)
    for n, sequence in enumerate(seqs):
        for j, char in enumerate(sequence):
            idx = aa_to_idx.get(char, aa_to_idx.get("*", 0))
            output[n, j * len(alphabet) + idx] = 1.0
    return output


def aac(seqs: List[str]) -> np.ndarray:
    aa_to_idx = {aa: i for i, aa in enumerate(AA20)}
    seq_len = len(seqs[0])
    output = np.zeros((len(seqs), 20), dtype=np.float32)
    for i, sequence in enumerate(seqs):
        for char in sequence:
            if char in aa_to_idx:
                output[i, aa_to_idx[char]] += 1.0
        output[i] /= float(seq_len)
    return output


def cksaap_k1(seqs: List[str], alphabet: List[str]) -> np.ndarray:
    alphabet_size = len(alphabet)
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    output = np.zeros((len(seqs), alphabet_size * alphabet_size), dtype=np.float32)
    for n, sequence in enumerate(seqs):
        ids = [aa_to_idx.get(char, aa_to_idx["X"]) for char in sequence]
        idx = np.array(ids[:-1], dtype=np.int64) * alphabet_size + np.array(ids[1:], dtype=np.int64)
        counts = np.bincount(idx, minlength=alphabet_size * alphabet_size).astype(np.float32)
        output[n] = counts
    return output


BLOSUM62_24 = """\
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def parse_blosum62_24() -> np.ndarray:
    lines = [line.strip() for line in BLOSUM62_24.strip().splitlines()]
    matrix = np.zeros((24, 24), dtype=np.float32)
    for i, line in enumerate(lines):
        parts = [int(value) for value in line.split()]
        if len(parts) != 24:
            raise ValueError("BLOSUM62_24 parsing failed: column count is not 24.")
        matrix[i] = np.array(parts, dtype=np.float32)
    return matrix


class GPSEncoder:
    alist = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "B", "Z", "X", "*"]

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.mm = parse_blosum62_24()
        self.count_matrix: Optional[np.ndarray] = None
        self.a2i = {aa: i for i, aa in enumerate(self.alist)}
        self.iu1 = np.triu_indices(24)

    def fit(self, seqs: List[str]) -> "GPSEncoder":
        counts = np.zeros((self.seq_len, 24), dtype=np.float64)
        for sequence in seqs:
            for pos, char in enumerate(sequence):
                aa = char if char in self.a2i else "X"
                counts[pos, self.a2i[aa]] += 1.0
        counts /= float(len(seqs))
        self.count_matrix = counts.astype(np.float32)
        return self

    def transform(self, seqs: List[str]) -> np.ndarray:
        if self.count_matrix is None:
            raise RuntimeError("GPSEncoder has not been fitted.")
        output = np.zeros((len(seqs), 300), dtype=np.float32)
        count_matrix = self.count_matrix
        for n, sequence in enumerate(seqs):
            indicator = np.zeros((self.seq_len, 24), dtype=np.float32)
            for pos, char in enumerate(sequence):
                aa = char if char in self.a2i else "X"
                indicator[pos, self.a2i[aa]] = 1.0
            matrix = (indicator.T @ count_matrix) * self.mm
            matrix = matrix + matrix.T
            np.fill_diagonal(matrix, np.diag(matrix) / 2.0)
            output[n] = matrix[self.iu1]
        return output


def build_feature_matrices(
    train_seqs: List[str],
    val_seqs: List[str],
    test_seqs: List[str],
    cfg,
    cache_dir: Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, object]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    preprocess: Dict[str, Dict[str, object]] = {}
    features: Dict[str, Dict[str, str]] = {}

    def save_array(feature_name: str, split: str, array: np.ndarray) -> str:
        path = cache_dir / f"{feature_name}_{split}.npy"
        np.save(path, array.astype(np.float32))
        return str(path)

    enabled = {
        "ACF": cfg.enable_acf,
        "AAINDEX": cfg.enable_aaindex,
        "OBC": cfg.enable_obc,
        "GPS": cfg.enable_gps,
        "CKSAAP": cfg.enable_cksaap,
        "PSEAAC": cfg.enable_pseaac,
    }

    if enabled["ACF"] or enabled["AAINDEX"]:
        aa_header, feat_values = load_top10()
        preprocess["top10"] = {"aa_header": aa_header, "feat_names": list(feat_values.keys())}
        tr_acf, tr_aa = aaindex_encode(train_seqs, aa_header, feat_values)
        va_acf, va_aa = aaindex_encode(val_seqs, aa_header, feat_values)
        te_acf, te_aa = aaindex_encode(test_seqs, aa_header, feat_values)
        if enabled["ACF"]:
            features["ACF"] = {
                "train": save_array("ACF", "train", tr_acf),
                "val": save_array("ACF", "val", va_acf),
                "test": save_array("ACF", "test", te_acf),
            }
        if enabled["AAINDEX"]:
            features["AAINDEX"] = {
                "train": save_array("AAINDEX", "train", tr_aa),
                "val": save_array("AAINDEX", "val", va_aa),
                "test": save_array("AAINDEX", "test", te_aa),
            }

    if enabled["OBC"]:
        obc_alphabet = ["K", "L", "A", "E", "V", "G", "S", "D", "I", "T", "R", "*", "P", "Q", "N", "F", "Y", "M", "H", "C", "W", "U"]
        preprocess["obc"] = {"alphabet": obc_alphabet}
        features["OBC"] = {
            "train": save_array("OBC", "train", obc_onehot_flat(train_seqs, obc_alphabet)),
            "val": save_array("OBC", "val", obc_onehot_flat(val_seqs, obc_alphabet)),
            "test": save_array("OBC", "test", obc_onehot_flat(test_seqs, obc_alphabet)),
        }

    if enabled["GPS"]:
        gps_encoder = GPSEncoder(seq_len=cfg.seq_len).fit(train_seqs)
        preprocess["gps"] = {"encoder": gps_encoder}
        features["GPS"] = {
            "train": save_array("GPS", "train", gps_encoder.transform(train_seqs)),
            "val": save_array("GPS", "val", gps_encoder.transform(val_seqs)),
            "test": save_array("GPS", "test", gps_encoder.transform(test_seqs)),
        }

    if enabled["CKSAAP"]:
        cksaap_alphabet = list("ACDEFGHIKLMNPQRSTVWXY")
        preprocess["cksaap"] = {"alphabet": cksaap_alphabet}
        features["CKSAAP"] = {
            "train": save_array("CKSAAP", "train", cksaap_k1(train_seqs, cksaap_alphabet)),
            "val": save_array("CKSAAP", "val", cksaap_k1(val_seqs, cksaap_alphabet)),
            "test": save_array("CKSAAP", "test", cksaap_k1(test_seqs, cksaap_alphabet)),
        }

    if enabled["PSEAAC"]:
        features["PSEAAC"] = {
            "train": save_array("PSEAAC", "train", aac(train_seqs)),
            "val": save_array("PSEAAC", "val", aac(val_seqs)),
            "test": save_array("PSEAAC", "test", aac(test_seqs)),
        }

    return features, preprocess


def build_feature_matrices_for_inference(
    seqs: List[str],
    cfg,
    cache_dir: Path,
    gps_encoder: Optional[GPSEncoder],
) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    features: Dict[str, str] = {}

    def save_array(feature_name: str, array: np.ndarray) -> str:
        path = cache_dir / f"{feature_name}_predict.npy"
        np.save(path, array.astype(np.float32))
        return str(path)

    if cfg.enable_acf or cfg.enable_aaindex:
        aa_header, feat_values = load_top10()
        acf_feature, aaindex_feature = aaindex_encode(seqs, aa_header, feat_values)
        if cfg.enable_acf:
            features["ACF"] = save_array("ACF", acf_feature)
        if cfg.enable_aaindex:
            features["AAINDEX"] = save_array("AAINDEX", aaindex_feature)

    if cfg.enable_obc:
        obc_alphabet = ["K", "L", "A", "E", "V", "G", "S", "D", "I", "T", "R", "*", "P", "Q", "N", "F", "Y", "M", "H", "C", "W", "U"]
        features["OBC"] = save_array("OBC", obc_onehot_flat(seqs, obc_alphabet))

    if cfg.enable_gps:
        if gps_encoder is None:
            raise RuntimeError("GPS encoder is required for HybridKla inference when GPS is enabled.")
        features["GPS"] = save_array("GPS", gps_encoder.transform(seqs))

    if cfg.enable_cksaap:
        cksaap_alphabet = list("ACDEFGHIKLMNPQRSTVWXY")
        features["CKSAAP"] = save_array("CKSAAP", cksaap_k1(seqs, cksaap_alphabet))

    if cfg.enable_pseaac:
        features["PSEAAC"] = save_array("PSEAAC", aac(seqs))

    return features
