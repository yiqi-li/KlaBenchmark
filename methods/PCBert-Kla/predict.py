from __future__ import annotations

import atexit
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import (
    SequencePhysDataset,
    build_collate_fn,
    compute_physchem_features,
    extract_sequences_and_labels,
    load_tsv,
    warn_raw_seq_len_adjustments,
)
from model import CLASSIFIER_INPUT_DIM, PCBertKlaClassifier, build_tokenizer_and_backbone
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    compute_metrics,
    ensure_dir,
    load_json,
    pick_device,
    save_json,
    save_predictions,
    timestamp,
    validate_prediction_alignment,
)


def load_training_defaults(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        return {}

    checkpoint = Path(checkpoint_path).resolve()
    output_root = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent

    overrides = {
        "checkpoint_path": str(checkpoint),
        "scaler_path": str(output_root / "model" / "extra" / "scaler_model.pkl"),
        "run_meta_path": str(output_root / "model" / "extra" / "run_meta.json"),
    }

    run_config_path = output_root / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        skip_keys = {"output_dir", "out_root", "run_name"}
        for key in Config.__dataclass_fields__:
            if key in run_config and key not in skip_keys:
                overrides[key] = run_config[key]

    preprocess_path = output_root / "preprocess.json"
    if preprocess_path.exists():
        preprocess = load_json(preprocess_path)
        if "seq_len" in preprocess:
            overrides["seq_len"] = preprocess["seq_len"]

    run_meta_path = output_root / "model" / "extra" / "run_meta.json"
    if run_meta_path.exists():
        run_meta = load_json(run_meta_path)
        protbert_meta = run_meta.get("protbert", {})
        path_meta = run_meta.get("paths", {})
        if "model_name" in protbert_meta:
            overrides["protbert_dir"] = protbert_meta["model_name"]
        if "keep_bert_layers" in protbert_meta:
            overrides["keep_bert_layers"] = protbert_meta["keep_bert_layers"]
        if "local_files_only" in protbert_meta:
            overrides["local_files_only"] = protbert_meta["local_files_only"]
        if "scaler_path" in path_meta:
            overrides["scaler_path"] = path_meta["scaler_path"]
        overrides["run_meta_path"] = str(run_meta_path)

    return overrides


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    checkpoint = Path(config.checkpoint_path).resolve()
    base_dir = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    return (base_dir / f"predict_{timestamp()}").resolve()


def load_model(config, cache_dir: Path, device: torch.device):
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    tokenizer, bert, max_length_tokens = build_tokenizer_and_backbone(config, cache_dir, device)

    input_dim = int(checkpoint.get("input_dim", CLASSIFIER_INPUT_DIM))
    classifier = PCBertKlaClassifier(input_dim=input_dim).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    bert.load_state_dict(checkpoint["protbert_state_dict"])

    scaler = joblib.load(config.scaler_path)
    classifier.eval()
    bert.eval()
    return tokenizer, bert, classifier, scaler, max_length_tokens


def predict_scores(classifier, bert, loader: DataLoader, device: torch.device, has_labels: bool):
    num_samples = len(loader.dataset)  # type: ignore[arg-type]
    probabilities = np.empty((num_samples,), dtype=np.float32)
    labels = np.empty((num_samples,), dtype=np.int64) if has_labels else None
    seen = np.zeros((num_samples,), dtype=bool)

    with torch.no_grad():
        for inputs, phys, batch_labels, indices in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            phys = phys.to(device)

            embeddings = bert(**inputs).last_hidden_state[:, 0, :]
            fused = torch.cat([embeddings, phys], dim=1)
            logits = classifier(fused)
            batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            index_np = indices.detach().cpu().numpy().astype(np.int64)

            if index_np.ndim != 1:
                raise RuntimeError(f"Unexpected idx shape: {index_np.shape}")
            if (index_np < 0).any() or (index_np >= num_samples).any():
                raise RuntimeError(
                    f"Index out of range: min={index_np.min()} max={index_np.max()} n={num_samples}"
                )
            if seen[index_np].any():
                duplicate = index_np[seen[index_np]][0]
                raise RuntimeError(f"Detected duplicated idx={int(duplicate)} during prediction.")

            seen[index_np] = True
            probabilities[index_np] = batch_probabilities
            if has_labels and labels is not None:
                labels[index_np] = batch_labels.detach().cpu().numpy().astype(np.int64)

    if not seen.all():
        missing = np.where(~seen)[0]
        raise RuntimeError(f"Prediction alignment failed. Missing sample indices: {missing[:10].tolist()}")

    return labels, probabilities


def predict() -> None:
    parser = make_predict_parser()
    args = parser.parse_args()
    provisional_config, file_overrides, cli_overrides = config_and_overrides_from_args(args)
    artifact_overrides = load_training_defaults(provisional_config.checkpoint_path)
    config = merge_config(Config(), artifact_overrides, file_overrides, cli_overrides)

    if not config.checkpoint_path:
        raise ValueError("--checkpoint_path is required for prediction.")
    if not config.test_tsv:
        raise ValueError("--test_tsv/--input_tsv is required for prediction.")
    if not config.scaler_path:
        raise ValueError("Unable to resolve scaler_path. Please provide --scaler_path explicitly.")

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[{METHOD_NAME}] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    device = pick_device(config.device)
    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)

    cache_dir = output_dir / "cache"
    predictions_path = output_dir / "predictions.tsv"
    metrics_path = output_dir / "predict_metrics.json"
    ensure_dir(output_dir)
    ensure_dir(cache_dir)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print(f"Loading checkpoint from: {config.checkpoint_path}")
    print(f"Input TSV: {config.test_tsv}")
    print(f"Output directory: {output_dir}")

    tokenizer, bert, classifier, scaler, max_length_tokens = load_model(config, cache_dir, device)

    input_df = load_tsv(Path(config.test_tsv))
    sequences, labels, seq_col = extract_sequences_and_labels(input_df, require_label=False)
    has_labels = labels is not None

    warn_raw_seq_len_adjustments(sequences, int(config.seq_len), split_label="predict")

    physchem = compute_physchem_features(sequences, seq_len=int(config.seq_len))
    physchem_scaled = scaler.transform(physchem).astype(np.float32)

    collate_fn = build_collate_fn(tokenizer, seq_len=int(config.seq_len), max_length_tokens=max_length_tokens)
    data_loader = DataLoader(
        SequencePhysDataset(sequences, physchem_scaled, labels),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    y_true, y_prob = predict_scores(classifier, bert, data_loader, device, has_labels=has_labels)
    input_df_used = input_df.iloc[: len(sequences)].copy()
    validate_prediction_alignment(input_df_used, seq_col, sequences)
    prediction_frame = build_prediction_dataframe(
        input_df_used,
        seq_col=seq_col,
        probabilities=y_prob,
        threshold=float(config.decision_threshold),
    )
    save_predictions(predictions_path, prediction_frame)

    save_json(output_dir / "predict_config.json", config.to_dict())

    if has_labels and y_true is not None:
        metrics = compute_metrics(y_true, y_prob, threshold=float(config.decision_threshold))
        save_json(
            metrics_path,
            {
                "method": METHOD_NAME,
                "primary_result": "prediction_only",
                "metrics": metrics,
                "paths": {
                    "checkpoint": config.checkpoint_path,
                    "predictions": str(predictions_path),
                },
            },
        )
        print(f"Saved metrics to: {metrics_path}")

    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    predict()
