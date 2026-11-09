from __future__ import annotations

import atexit
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow

from configs import Config, METHOD_NAME, config_and_overrides_from_args, make_predict_parser, merge_config
from dataset import VOCAB_SIZE, encode_pair, ensure_with_contact_tsv, load_split_tsv
from utils import (
    TeeRunLog,
    build_standard_metrics_payload,
    build_standard_prediction_df,
    compute_metrics,
    ensure_dir,
    load_json,
    now_ts,
    pick_device,
    save_json,
    save_predictions,
    timestamp,
)


def _configure_tf_runtime(config):
    import tensorflow as tf

    if config.gpu:
        all_gpus = tf.config.list_physical_devices("GPU")
        if not all_gpus:
            print("[ABFF-Kla] WARN: no visible GPU found; --gpu will be ignored.")
        else:
            gpu_ids = [int(x.strip()) for x in str(config.gpu).split(",") if x.strip()]
            try:
                selected_gpus = [all_gpus[i] for i in gpu_ids]
            except IndexError as exc:
                raise ValueError(
                    f"--gpu exceeds the visible GPU range. visible_gpu_count={len(all_gpus)} received={config.gpu!r}"
                ) from exc
            tf.config.set_visible_devices(selected_gpus, "GPU")
            print(f"[ABFF-Kla] Visible GPU ids set to: {config.gpu}")

    if pick_device(config.device) != "cpu":
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return tf


def _infer_output_dir(config: Config) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    root = Path(config.out_root)
    run_name = config.run_name or f"predict_{timestamp()}"
    return root / run_name


def load_training_defaults(config: Config) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    candidate_files = []
    if config.run_meta_path:
        candidate_files.append(Path(config.run_meta_path))
    if checkpoint_path is not None:
        candidate_files.append(checkpoint_path.parent / "extra" / "run_meta.json")
        candidate_files.append(checkpoint_path.parent.parent / "run_config.json")

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if path.name == "run_config.json":
            for key, value in payload.items():
                if key in {"method", "dataset_name", "output_dir", "out_root", "run_name"}:
                    continue
                defaults[key] = value
            continue

        config_payload = payload.get("config")
        if isinstance(config_payload, dict):
            for key, value in config_payload.items():
                if key in {"output_dir", "out_root", "run_name"}:
                    continue
                defaults[key] = value
        for meta_key, target_key in [
            ("best_model_path", "checkpoint_path"),
            ("preprocess_path", "preprocess_path"),
        ]:
            value = payload.get(meta_key)
            if value:
                defaults[target_key] = value
    return defaults


def main() -> int:
    parser = make_predict_parser()
    args = parser.parse_args()
    config, _, cli_overrides = config_and_overrides_from_args(args)
    training_defaults = load_training_defaults(config)
    config = merge_config(config, training_defaults, cli_overrides)

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required for prediction.")

    tf = _configure_tf_runtime(config)
    from model import load_trained_model

    output_dir = _infer_output_dir(config)
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache" / "derived_contact"
    ensure_dir(output_dir)
    ensure_dir(results_dir)
    ensure_dir(cache_dir)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print(f"[{now_ts()}] Prediction output dir: {output_dir}")

    preprocess_path = Path(config.preprocess_path) if config.preprocess_path else None
    if preprocess_path is None or not preprocess_path.exists():
        candidate = Path(config.checkpoint_path).parent.parent / "preprocess.json"
        if candidate.exists():
            preprocess_path = candidate
    if preprocess_path is None or not preprocess_path.exists():
        raise FileNotFoundError("preprocess_path is required and could not be inferred from checkpoint_path.")
    preprocess = load_json(preprocess_path)

    test_tsv = ensure_with_contact_tsv(
        config.test_tsv,
        run_dir=cache_dir,
        pdb_dir=config.pdb_dir,
        cache_dir=config.contact_cache_dir,
        chain_id=config.chain_id,
        cutoff_a=config.contact_cutoff,
        seg_len=int(preprocess["L"]),
        report_every=config.report_every,
        reuse_cache=(not config.no_reuse_contact_cache),
        force_regen=config.force_regen_contact,
    )

    df_input, input_ctx, input_cont, input_labels = load_split_tsv(test_tsv, require_label=False)
    x_acid, x_cont, empty_contact = encode_pair(
        input_ctx, input_cont, length=int(preprocess["L"]), split_label="predict"
    )

    model = load_trained_model(config.checkpoint_path)
    y_prob = model.predict([x_acid, x_cont], batch_size=int(config.batch_size), verbose=0).reshape(-1)
    pred_df = build_standard_prediction_df(df_input.reset_index(drop=True), y_prob, float(config.decision_threshold), "Sequence")
    save_predictions(results_dir / "test_predictions.tsv", pred_df)

    predict_meta: dict[str, Any] = {
        "method": METHOD_NAME,
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "preprocess_path": str(preprocess_path.resolve()),
        "input_tsv": str(Path(config.test_tsv).resolve()),
        "prediction_input_tsv": str(Path(test_tsv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "prediction_path": str((results_dir / "test_predictions.tsv").resolve()),
        "empty_contact_count": int(empty_contact),
        "config": asdict(config),
        "vocab_size": int(preprocess.get("vocab_size", VOCAB_SIZE)),
    }

    if input_labels is not None:
        metrics = compute_metrics(input_labels, y_prob, threshold=float(config.decision_threshold))
        metrics_payload = build_standard_metrics_payload(
            method=METHOD_NAME,
            y_true=input_labels,
            y_prob=y_prob,
            raw_metrics=metrics,
            threshold=float(config.decision_threshold),
            primary_result="checkpoint",
            monitor="provided_checkpoint",
            best_epoch=None,
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath=Path(config.checkpoint_path).name,
        )
        save_json(results_dir / "test_metrics.json", metrics_payload)
        predict_meta["metrics_path"] = str((results_dir / "test_metrics.json").resolve())

    save_json(output_dir / "predict_config.json", predict_meta)
    print(f"[{now_ts()}] Prediction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
