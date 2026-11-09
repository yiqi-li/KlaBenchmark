from __future__ import annotations

import atexit
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import AA_TO_ID, VOCAB_SIZE, encode_pair, ensure_with_contact_tsv, load_split_tsv
from utils import (
    TeeRunLog,
    build_run_paths,
    build_standard_metrics_payload,
    build_standard_prediction_df,
    compute_metrics,
    ensure_dir,
    now_ts,
    pick_device,
    save_json,
    save_predictions,
    set_seed,
    summarize_binary_split,
    timestamp,
)


def _default_run_name(train_tsv: str) -> str:
    dataset_name = Path(train_tsv).resolve().parent.name or "train"
    return f"{dataset_name}_{timestamp()}"


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


def main() -> int:
    parser = make_train_parser()
    args = parser.parse_args()
    config, file_overrides, cli_overrides = config_and_overrides_from_args(args)

    tf = _configure_tf_runtime(config)
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

    from model import ABFF_Kla_model, PrettyLogger, ensure_pretrained_models, load_trained_model

    set_seed(int(config.seed))

    run_name = config.run_name or _default_run_name(config.train_tsv)
    output_dir = Path(config.output_dir) if config.output_dir else Path(config.out_root) / run_name
    paths = build_run_paths(output_dir)
    for path in [
        paths.output_dir,
        paths.model_dir,
        paths.model_extra_dir,
        paths.checkpoint_dir,
        paths.results_dir,
        paths.cache_dir,
        paths.derived_contact_dir,
    ]:
        ensure_dir(path)

    run_log = TeeRunLog(paths.output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    dataset_name = Path(config.train_tsv).resolve().parent.name or "train"
    save_json(paths.run_config_path, {**asdict(config), "method": METHOD_NAME, "dataset_name": dataset_name})

    train_tsv = ensure_with_contact_tsv(
        config.train_tsv,
        run_dir=paths.derived_contact_dir,
        pdb_dir=config.pdb_dir,
        cache_dir=config.contact_cache_dir,
        chain_id=config.chain_id,
        cutoff_a=config.contact_cutoff,
        seg_len=config.seq_len,
        report_every=config.report_every,
        reuse_cache=(not config.no_reuse_contact_cache),
        force_regen=config.force_regen_contact,
    )
    val_tsv = ensure_with_contact_tsv(
        config.val_tsv,
        run_dir=paths.derived_contact_dir,
        pdb_dir=config.pdb_dir,
        cache_dir=config.contact_cache_dir,
        chain_id=config.chain_id,
        cutoff_a=config.contact_cutoff,
        seg_len=config.seq_len,
        report_every=config.report_every,
        reuse_cache=(not config.no_reuse_contact_cache),
        force_regen=config.force_regen_contact,
    )
    test_tsv = ensure_with_contact_tsv(
        config.test_tsv,
        run_dir=paths.derived_contact_dir,
        pdb_dir=config.pdb_dir,
        cache_dir=config.contact_cache_dir,
        chain_id=config.chain_id,
        cutoff_a=config.contact_cutoff,
        seg_len=config.seq_len,
        report_every=config.report_every,
        reuse_cache=(not config.no_reuse_contact_cache),
        force_regen=config.force_regen_contact,
    )

    preprocess = {
        "L": int(config.seq_len),
        "vocab_size": int(VOCAB_SIZE),
        "aa_to_id": AA_TO_ID,
        "pad_char": "_",
        "empty_contact_policy": "empty -> all '_'",
    }
    save_json(paths.preprocess_path, preprocess)

    df_train, train_ctx, train_cont, y_train = load_split_tsv(train_tsv, require_label=True)
    df_val, val_ctx, val_cont, y_val = load_split_tsv(val_tsv, require_label=True)
    df_test, test_ctx, test_cont, y_test = load_split_tsv(test_tsv, require_label=True)
    assert y_train is not None and y_val is not None and y_test is not None

    X_train_acid, X_train_cont, train_empty = encode_pair(
        train_ctx, train_cont, length=int(config.seq_len), split_label="train"
    )
    X_val_acid, X_val_cont, val_empty = encode_pair(
        val_ctx, val_cont, length=int(config.seq_len), split_label="val"
    )
    X_test_acid, X_test_cont, test_empty = encode_pair(
        test_ctx, test_cont, length=int(config.seq_len), split_label="test"
    )

    print("\n========== DATA SUMMARY ==========")
    print(
        f"Train: {len(y_train)} | positives={int(y_train.sum())} | negatives={int((1 - y_train).sum())} | empty_contact={train_empty}"
    )
    print(
        f"Val:   {len(y_val)}   | positives={int(y_val.sum())}   | negatives={int((1 - y_val).sum())}   | empty_contact={val_empty}"
    )
    print(
        f"Test:  {len(y_test)}  | positives={int(y_test.sum())}  | negatives={int((1 - y_test).sum())}  | empty_contact={test_empty}"
    )
    print("=================================\n")
    save_json(
        paths.dataset_stats_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "train": summarize_binary_split("train", y_train, empty_contact=train_empty),
            "val": summarize_binary_split("val", y_val, empty_contact=val_empty),
            "test": summarize_binary_split("test", y_test, empty_contact=test_empty),
        },
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.lr))
    model = ABFF_Kla_model(
        L=int(config.seq_len),
        vocab_size=int(VOCAB_SIZE),
        embedding_size=int(config.embed),
        attention_size=int(config.attn),
        lstm_units=int(config.lstm),
        dense_size=int(config.dense),
        optimizer=optimizer,
    )
    model.summary()

    acid_w, contmap_w = ensure_pretrained_models(
        run_dir=paths.model_extra_dir,
        X_train_acid=X_train_acid,
        X_train_cont=X_train_cont,
        y_train=y_train,
        X_val_acid=X_val_acid,
        X_val_cont=X_val_cont,
        y_val=y_val,
        L=int(config.seq_len),
        vocab_size=int(VOCAB_SIZE),
        embedding_size=int(config.embed),
        attention_size=int(config.attn),
        lstm_units=int(config.lstm),
        dense_size=int(config.dense),
        lr=float(config.lr),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        early_patience=int(config.early_patience),
        reduce_patience=int(config.reduce_patience),
    )
    model.load_weights(str(acid_w), by_name=True)
    model.load_weights(str(contmap_w), by_name=True)

    callbacks = [
        PrettyLogger(train_n=len(y_train), val_n=len(y_val)),
        ModelCheckpoint(
            filepath=str(paths.best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        CSVLogger(str(paths.training_log_path), append=False),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=int(config.early_patience),
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x=[X_train_acid, X_train_cont],
        y=y_train,
        validation_data=([X_val_acid, X_val_cont], y_val),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )

    model.save(str(paths.final_model_path))
    print(f"\nSaved final model to: {paths.final_model_path}")
    print(f"Saved best model  to: {paths.best_model_path}")

    best_model = load_trained_model(paths.best_model_path)
    try:
        best_epoch = int(np.nanargmax(np.asarray(history.history.get("val_auc", []), dtype=float)) + 1)
    except Exception:
        best_epoch = None

    decision_cutoff = float(config.decision_threshold)
    y_val_prob = best_model.predict([X_val_acid, X_val_cont], batch_size=int(config.batch_size), verbose=0).reshape(-1)
    val_metrics_out = compute_metrics(y_val, y_val_prob, threshold=decision_cutoff)

    print("\n========== TEST INFERENCE & EVAL ==========")
    y_prob = best_model.predict([X_test_acid, X_test_cont], batch_size=int(config.batch_size), verbose=0).reshape(-1)
    metrics_out = compute_metrics(y_test, y_prob, threshold=decision_cutoff)
    for key in ["auc", "prauc", "acc", "precision", "recall", "f1", "mcc"]:
        value = metrics_out[key]
        print(f"{key:10s}: {value:.6f}" if isinstance(value, (float, np.floating)) else f"{key:10s}: {value}")

    df_test_out = df_test.copy().reset_index(drop=True)
    if len(df_test_out) != len(y_prob):
        raise RuntimeError(f"Alignment check failed: len(df_test)={len(df_test_out)} != len(y_prob)={len(y_prob)}")

    sig_df = (
        df_test_out["Sequence"].fillna("").astype(str).str.upper().str.strip()
        + "||"
        + df_test_out["Contact"].fillna("").astype(str).str.upper().str.strip()
        + "||"
        + df_test_out["Label"].astype(str)
    ).to_numpy()
    sig_used = (
        pd.Series(test_ctx).fillna("").astype(str).str.upper().str.strip()
        + "||"
        + pd.Series(test_cont).fillna("").astype(str).str.upper().str.strip()
        + "||"
        + pd.Series(y_test).astype(str)
    ).to_numpy()
    if not np.array_equal(sig_df, sig_used):
        mismatch = np.where(sig_df != sig_used)[0]
        first = int(mismatch[0]) if len(mismatch) > 0 else -1
        raise RuntimeError(
            "Alignment check failed: test dataframe order differs from prediction input order.\n"
            f"first_mismatch_index={first}\n"
            f"df_sig={sig_df[first] if first >= 0 else 'NA'}\n"
            f"used_sig={sig_used[first] if first >= 0 else 'NA'}"
        )

    pred_df = build_standard_prediction_df(df_test_out, y_prob, decision_cutoff, sequence_col="Sequence")
    save_predictions(paths.test_predictions_path, pred_df)

    val_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        y_true=y_val,
        y_prob=y_val_prob,
        raw_metrics=val_metrics_out,
        threshold=decision_cutoff,
        primary_result="best",
        monitor="val_auroc",
        best_epoch=best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.keras",
    )
    test_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        y_true=y_test,
        y_prob=y_prob,
        raw_metrics=metrics_out,
        threshold=decision_cutoff,
        primary_result="best",
        monitor="val_auroc",
        best_epoch=best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.keras",
    )
    save_json(paths.val_metrics_path, val_payload)
    save_json(paths.test_metrics_path, test_payload)

    print(f"Saved val metrics to: {paths.val_metrics_path}")
    print(f"Saved test metrics to: {paths.test_metrics_path}")
    print(f"Saved test predictions to: {paths.test_predictions_path}")
    print("==========================================\n")

    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "run_name": run_name,
            "output_dir": str(paths.output_dir),
            "best_model_path": str(paths.best_model_path),
            "final_model_path": str(paths.final_model_path),
            "acid_pretrain_path": str(paths.acid_pretrain_path),
            "contact_pretrain_path": str(paths.contact_pretrain_path),
            "preprocess_path": str(paths.preprocess_path),
            "run_config_path": str(paths.run_config_path),
            "train_with_contact_tsv": str(train_tsv),
            "val_with_contact_tsv": str(val_tsv),
            "test_with_contact_tsv": str(test_tsv),
            "decision_threshold": decision_cutoff,
            "best_epoch": best_epoch,
            "config": asdict(config),
            "config_file_overrides": file_overrides,
            "config_cli_overrides": cli_overrides,
        },
    )
    print(f"[{now_ts()}] Training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
