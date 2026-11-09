from __future__ import annotations

import atexit
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import BenchmarkDataset, build_embedding_inputs, load_tsv, make_features_for_split
from model import DeepKlapredLogits
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    compute_metrics,
    ensure_dir,
    load_json,
    now_ts,
    pick_device,
    run_predict_proba_with_index,
    save_json,
    save_predictions,
    timestamp,
)


def load_training_defaults(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        return {}
    checkpoint = Path(checkpoint_path).resolve()
    output_root = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    overrides = {
        "checkpoint_path": str(checkpoint),
        "preprocess_path": str(output_root / "preprocess.json"),
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
        overrides["preprocess_path"] = str(preprocess_path)

    run_meta_path = output_root / "model" / "extra" / "run_meta.json"
    if run_meta_path.exists():
        run_meta = load_json(run_meta_path)
        config_meta = run_meta.get("config", {})
        for key in Config.__dataclass_fields__:
            if key in config_meta and key not in {"output_dir", "out_root", "run_name"}:
                overrides[key] = config_meta[key]
        overrides["run_meta_path"] = str(run_meta_path)

    return overrides


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    checkpoint = Path(config.checkpoint_path).resolve()
    base_dir = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    return (base_dir / f"predict_{timestamp()}").resolve()


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

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[DeepKlapred] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    ensure_dir(output_dir)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    device = pick_device(config.device)
    preprocess = load_json(Path(config.preprocess_path)) if config.preprocess_path else {}

    input_df, input_seqs, input_y = load_tsv(config.test_tsv, require_label=False)
    max_seq_len = int(preprocess.get("max_seq_len", config.seq_len))
    max_len_with_cls = int(preprocess.get("max_len_with_cls", max_seq_len + 1))
    feature_columns = preprocess.get("feature_columns", None)

    print(f"[{now_ts()}] Device: {device}")
    print(f"[{now_ts()}] Building inputs for prediction...")
    x_ids = build_embedding_inputs(input_seqs, max_len_with_cls, split_label="predict")
    x_desc, feature_columns = make_features_for_split(
        input_seqs, feature_columns=feature_columns, raw_seq_len=max_seq_len
    )

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model = DeepKlapredLogits(
        vocab_size=int(checkpoint.get("vocab_size", 24)),
        seq_feature_dim=int(checkpoint.get("seq_feature_dim", x_desc.shape[1])),
        max_len=int(checkpoint.get("max_len", max_len_with_cls)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels = input_y if input_y is not None else np.zeros((len(input_seqs),), dtype=np.int64)
    dataset = BenchmarkDataset(x_ids, x_desc, labels)
    loader = Data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    _idx, y_true_aligned, y_prob_pos_aligned = run_predict_proba_with_index(model, loader, device)
    assert len(input_df) == int(y_prob_pos_aligned.shape[0]), (
        f"Input TSV rows != predictions: tsv={len(input_df)} preds={int(y_prob_pos_aligned.shape[0])}"
    )
    if input_y is not None:
        assert np.array_equal(input_df["Label"].astype(int).to_numpy(), y_true_aligned), "Label mismatch between TSV order and dataloader order"

    prediction_df = build_prediction_dataframe(input_df, y_prob_pos_aligned, threshold=float(config.decision_threshold))
    predictions_path = output_dir / "predictions.tsv"
    save_predictions(predictions_path, prediction_df)
    save_json(output_dir / "predict_config.json", config.to_dict())

    if input_y is not None:
        metrics = compute_metrics(y_true_aligned, y_prob_pos_aligned, threshold=float(config.decision_threshold))
        save_json(
            output_dir / "predict_metrics.json",
            {
                "method": METHOD_NAME,
                "primary_result": "prediction_only",
                "metrics": metrics.__dict__,
                "paths": {
                    "checkpoint": config.checkpoint_path,
                    "predictions": str(predictions_path),
                },
            },
        )

    print(f"[{now_ts()}] Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    predict()
