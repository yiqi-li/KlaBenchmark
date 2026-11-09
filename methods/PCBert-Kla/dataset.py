from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from torch.utils.data import Dataset

from utils import align_raw_sequence, log_seq_len_adjustments, raw_sequence_length

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def load_tsv(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [column.strip() for column in df.columns]
    return df


def extract_sequences_and_labels(
    df: pd.DataFrame,
    seq_col_candidates: Sequence[str] = ("Sequence", "Context_31aa"),
    require_label: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    seq_col = None
    for column in seq_col_candidates:
        if column in df.columns:
            seq_col = column
            break
    if seq_col is None:
        raise ValueError(f"Missing sequence column. Candidates={tuple(seq_col_candidates)}, columns={list(df.columns)}")

    labels: np.ndarray | None = None
    if "Label" in df.columns:
        labels = df["Label"].astype(int).to_numpy()
    elif require_label:
        raise ValueError(f"Missing Label column. Columns={list(df.columns)}")

    sequences = df[seq_col].astype(str).to_numpy()
    return sequences, labels, seq_col


def sanitize_for_embedding(seq: str) -> str:
    """Project raw sequence tokens to the alphabet accepted by ProtBert."""
    sequence = (seq or "").strip().replace(" ", "").upper()
    output = []
    for token in sequence:
        if token in STANDARD_AA or token == "X":
            output.append(token)
        elif token == "_":
            output.append("X")
        else:
            output.append("X")
    return "".join(output)


def _normalize_raw_sequence_chars(sequence: str) -> str:
    sequence = sequence.replace(" ", "")
    output = []
    for token in sequence:
        if token == "_" or token in STANDARD_AA or token == "X":
            output.append(token)
        else:
            output.append("X")
    return "".join(output)


def normalize_seq_len_raw(seq: str, seq_len: int, pad_char: str = "_") -> str:
    if seq_len <= 0:
        sequence = (seq or "").strip().replace(" ", "").upper()
        return _normalize_raw_sequence_chars(sequence)
    aligned, _ = align_raw_sequence(
        seq,
        seq_len,
        pad_char=pad_char,
        normalize_chars=_normalize_raw_sequence_chars,
    )
    return aligned


def warn_raw_seq_len_adjustments(
    seqs: Iterable[str],
    seq_len: int,
    *,
    method_label: str = "PCBert-Kla",
    split_label: str = "",
) -> None:
    log_seq_len_adjustments(
        [raw_sequence_length(seq) for seq in seqs],
        int(seq_len),
        method_label=method_label,
        split_label=split_label,
    )


def sanitize_for_physchem(seq: str) -> str:
    """Keep only standard amino acids for ProteinAnalysis feature extraction."""
    sequence = (seq or "").strip().replace(" ", "").upper()
    sequence = "".join(token for token in sequence if token in STANDARD_AA)
    return sequence if sequence else "A"


def seq_to_protbert_tokens(seq: str) -> str:
    return " ".join(list(seq))


def compute_physchem_features(seqs: Iterable[str], seq_len: int) -> np.ndarray:
    """
    Return shape [N, 27]:
    molecular_weight(1), isoelectric_point(1), AAC(20),
    secondary_structure_fraction(3), gravy(1), charge_at_pH_7(1).
    """
    features = []
    for raw_seq in seqs:
        normalized = normalize_seq_len_raw(raw_seq, seq_len=int(seq_len))
        physchem_seq = sanitize_for_physchem(normalized)
        analysis = ProteinAnalysis(physchem_seq)
        molecular_weight = analysis.molecular_weight()
        isoelectric_point = analysis.isoelectric_point()
        aa_percent = getattr(analysis, "amino_acids_percent", None)
        if aa_percent is None:
            aa_percent = analysis.get_amino_acids_percent()
        amino_acid_composition = [float(aa_percent.get(amino_acid, 0.0)) for amino_acid in AA_ORDER]
        secondary_structure = list(analysis.secondary_structure_fraction())
        gravy = analysis.gravy()
        charge_at_ph_7 = analysis.charge_at_pH(7.0)
        features.append(
            [
                molecular_weight,
                isoelectric_point,
                *amino_acid_composition,
                *secondary_structure,
                gravy,
                charge_at_ph_7,
            ]
        )
    return np.asarray(features, dtype=np.float32)


class SequencePhysDataset(Dataset):
    def __init__(self, seqs_raw: np.ndarray, phys_arr: np.ndarray, labels: np.ndarray | None = None):
        if labels is not None and not (len(seqs_raw) == len(phys_arr) == len(labels)):
            raise ValueError("Sequence, feature, and label lengths must match.")
        if labels is None and len(seqs_raw) != len(phys_arr):
            raise ValueError("Sequence and feature lengths must match.")
        self.seqs = seqs_raw.astype(str)
        self.phys = phys_arr.astype(np.float32)
        self.labels = None if labels is None else labels.astype(np.float32)

    def __len__(self) -> int:
        return int(self.seqs.shape[0])

    def __getitem__(self, idx: int):
        label = 0.0 if self.labels is None else float(self.labels[idx])
        return self.seqs[idx], self.phys[idx], label, int(idx)


def build_collate_fn(tokenizer, seq_len: int, max_length_tokens: int):
    def collate_fn(batch):
        seqs_raw, phys_arr, labels, idxs = zip(*batch)
        seqs_norm = [normalize_seq_len_raw(seq, seq_len=int(seq_len)) for seq in seqs_raw]
        seqs_clean = [sanitize_for_embedding(seq) for seq in seqs_norm]
        seqs_tok = [seq_to_protbert_tokens(seq) for seq in seqs_clean]
        inputs = tokenizer(
            seqs_tok,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length_tokens),
        )
        phys = torch.tensor(np.asarray(phys_arr, dtype=np.float32))
        label_tensor = torch.tensor(np.asarray(labels, dtype=np.float32))
        idx_tensor = torch.tensor(np.asarray(idxs, dtype=np.int64))
        return inputs, phys, label_tensor, idx_tensor

    return collate_fn
