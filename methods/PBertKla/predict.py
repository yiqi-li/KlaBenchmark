from __future__ import annotations

import atexit
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import encode_sequences, load_tsv, read_raw_tsv, validate_prediction_scores
from model import (
    build_finetuning_components,
    build_inference_model,
    configure_tensorflow_environment,
    configure_tensorflow_memory_growth,
    import_tensorflow_or_raise,
    load_encoder_artifacts,
)
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    compute_metrics,
    default_run_name,
    ensure_dir,
    load_json,
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
        "run_meta_path": str(output_root / "model" / "extra" / "run_meta.json"),
        "input_encoder_path": str(output_root / "model" / "extra" / "input_encoder.pkl"),
        "output_spec_path": str(output_root / "model" / "extra" / "output_spec.pkl"),
    }

    run_config_path = output_root / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        skip_keys = {"output_dir", "out_root", "run_name"}
        for key in Config.__dataclass_fields__:
            if key in run_config and key not in skip_keys:
                overrides[key] = run_config[key]

    run_meta_path = output_root / "model" / "extra" / "run_meta.json"
    if run_meta_path.exists():
        run_meta = load_json(run_meta_path)
        path_meta = run_meta.get("paths", {})
        config_meta = run_meta.get("config", {})
        for key in Config.__dataclass_fields__:
            if key in config_meta and key not in {"output_dir", "out_root", "run_name"}:
                overrides[key] = config_meta[key]
        if "input_encoder_path" in path_meta:
            overrides["input_encoder_path"] = path_meta["input_encoder_path"]
        if "output_spec_path" in path_meta:
            overrides["output_spec_path"] = path_meta["output_spec_path"]
        if "run_meta_path" in path_meta:
            overrides["run_meta_path"] = path_meta["run_meta_path"]

    return overrides


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    checkpoint = Path(config.checkpoint_path).resolve()
    base_dir = checkpoint.parent.parent if checkpoint.parent.name == "model" else checkpoint.parent
    return (base_dir / f"predict_{timestamp()}").resolve()


def load_model(config, script_dir: Path):
    _, _, built_input_encoder, model_generator = build_finetuning_components(config, script_dir)
    input_encoder = built_input_encoder
    if config.input_encoder_path and Path(config.input_encoder_path).exists():
        try:
            input_encoder, _ = load_encoder_artifacts(Path(config.input_encoder_path), Path(config.output_spec_path))
        except Exception as exc:
            print(f"[PBertKla] WARN: failed to load pickled encoder artifacts, fallback to rebuilt encoder: {exc}")
    model = build_inference_model(model_generator, config.seq_len, config.checkpoint_path)
    return input_encoder, model


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

    configure_tensorflow_environment()
    if config.gpu:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
            print(f"[PBertKla] CUDA_VISIBLE_DEVICES not set; set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print(f"[PBertKla] CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']} (will not override)")
    else:
        print("[PBertKla] CUDA_VISIBLE_DEVICES not set by script (all GPUs visible).")
    configure_tensorflow_memory_growth(import_tensorflow_or_raise())

    script_dir = Path(__file__).resolve().parent
    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)

    ensure_dir(output_dir)
    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print(f"[PBertKla] Loading checkpoint from: {config.checkpoint_path}")
    print(f"[PBertKla] Input TSV: {config.test_tsv}")
    print(f"[PBertKla] Output dir: {output_dir}")

    input_encoder, model = load_model(config, script_dir)

    raw_input_df = read_raw_tsv(Path(config.test_tsv))
    input_df = load_tsv(Path(config.test_tsv), require_label=False)
    if len(raw_input_df) != len(input_df):
        raise RuntimeError(
            "[PBertKla][FATAL] Raw input TSV and validated input TSV have different row counts, "
            "so prediction output alignment is not safe."
        )

    X_input, _ = encode_sequences(
        input_encoder,
        input_df["Sequence"],
        config.seq_len,
        split_label="predict",
    )
    y_score = model.predict(X_input, batch_size=int(config.batch_size)).reshape(-1)
    validate_prediction_scores(y_score, len(input_df))

    predictions_path = output_dir / "predictions.tsv"
    prediction_df = build_prediction_dataframe(raw_input_df, y_score, float(config.decision_threshold))
    save_predictions(predictions_path, prediction_df)
    save_json(output_dir / "predict_config.json", config.to_dict())

    if "Label" in input_df.columns:
        metrics = compute_metrics(
            input_df["Label"].to_numpy().astype(int),
            y_score,
            threshold=float(config.decision_threshold),
        )
        save_json(
            output_dir / "predict_metrics.json",
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

    print(f"[PBertKla] Saved predictions to: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(predict())
