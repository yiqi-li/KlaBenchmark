from __future__ import annotations

import atexit
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import encode_sequences, load_tsv_rows
from model import configure_tensorflow_memory_growth, load_model
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    compute_metrics,
    ensure_dir,
    load_json,
    now_ts,
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
        overrides["preprocess_path"] = str(preprocess_path)
        preprocess = load_json(preprocess_path)
        if "seq_len" in preprocess:
            overrides["seq_len"] = preprocess["seq_len"]
    return overrides


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    checkpoint = Path(config.checkpoint_path).resolve()
    base_dir = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    return (base_dir / f"predict_{timestamp()}").resolve()


def predict() -> int:
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
        print(f"[DeepKla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    configure_tensorflow_memory_growth()

    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    ensure_dir(output_dir)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    preprocess = load_json(Path(config.preprocess_path)) if config.preprocess_path else {}
    seq_len = int(preprocess.get("seq_len", config.seq_len))
    vocab = preprocess.get("vocab", None)
    if not vocab:
        raise ValueError("Could not resolve vocab from preprocess.json.")

    rows, seqs, labels, fieldnames = load_tsv_rows(Path(config.test_tsv), require_label=False)
    x = encode_sequences(seqs, vocab=vocab, fixed_len=seq_len, split_label="predict")

    print(f"[{now_ts()}] Loading model from: {config.checkpoint_path}", flush=True)
    model = load_model(config.checkpoint_path)
    y_prob = model.predict(x, batch_size=config.batch_size, verbose=0).reshape(-1)
    if int(y_prob.shape[0]) != int(len(rows)):
        raise RuntimeError(f"Prediction count mismatch: pred={int(y_prob.shape[0])} vs rows={int(len(rows))}")

    extra_cols = [column for column in fieldnames if column not in {"Sequence", "Label"}]
    pred_df = build_prediction_dataframe(rows, extra_cols, y_prob, threshold=float(config.decision_threshold))
    pred_path = output_dir / "predictions.tsv"
    save_predictions(pred_path, pred_df)
    save_json(output_dir / "predict_config.json", config.to_dict())

    if labels is not None:
        metrics = compute_metrics(labels, y_prob, threshold=float(config.decision_threshold))
        save_json(
            output_dir / "predict_metrics.json",
            {
                "method": METHOD_NAME,
                "primary_result": "prediction_only",
                "metrics": metrics,
                "paths": {
                    "checkpoint": config.checkpoint_path,
                    "predictions": str(pred_path),
                },
            },
        )

    print(f"[{now_ts()}] Predictions saved to: {pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(predict())
