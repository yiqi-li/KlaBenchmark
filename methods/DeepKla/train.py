from __future__ import annotations

import atexit
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import build_vocab, clean_sequence, encode_sequences, load_tsv_rows, take_first
from model import build_model, build_rmsprop, configure_tensorflow_memory_growth
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    now_ts,
    save_json,
    save_predictions,
    set_seed,
    split_stats,
)


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    run_name = config.run_name or default_run_name(config.train_tsv)
    config.run_name = run_name
    return (Path(config.out_root) / run_name).resolve()


class ValMetricsCallback:
    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, batch_size: int, every_n_epochs: int = 1):
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = int(batch_size)
        self.every_n_epochs = every_n_epochs

    def to_keras_callback(self):
        from tensorflow import keras

        parent = self

        class _Callback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % parent.every_n_epochs != 0:
                    return
                y_prob = self.model.predict(parent.x_val, batch_size=parent.batch_size, verbose=0).reshape(-1)
                metrics = compute_metrics(parent.y_val, y_prob, threshold=0.5)
                print(
                    f"[{now_ts()}] val(extra): thr=0.5000 "
                    f"mcc={metrics['mcc']:.4f} f1={metrics['f1']:.4f} sn={metrics['sn']:.4f} sp={metrics['sp']:.4f} "
                    f"tp={int(metrics['tp'])} tn={int(metrics['tn'])} fp={int(metrics['fp'])} fn={int(metrics['fn'])}",
                    flush=True,
                )

        return _Callback()


def train() -> int:
    parser = make_train_parser()
    args = parser.parse_args()
    config, _, _ = config_and_overrides_from_args(args)

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[DeepKla] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    configure_tensorflow_memory_growth()

    set_seed(config.seed)
    if int(config.seq_len) <= 0:
        raise ValueError(f"--seq_len must be a positive integer. Got {config.seq_len!r}")

    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)
    dataset_name = Path(config.train_tsv).resolve().parent.name or "runs"
    paths = build_run_paths(output_dir)
    for directory in [paths.output_dir, paths.model_dir, paths.model_extra_dir, paths.results_dir, paths.cache_dir]:
        ensure_dir(directory)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    save_json(
        paths.run_config_path,
        {**config.to_dict(), "method": METHOD_NAME, "dataset_name": dataset_name},
    )

    print(f"[{now_ts()}] DeepKla{int(config.seq_len)} one-click run", flush=True)
    print(f"[{now_ts()}] output_dir: {output_dir}", flush=True)
    print(f"[{now_ts()}] train_tsv: {config.train_tsv}", flush=True)
    print(f"[{now_ts()}] val_tsv:   {config.val_tsv}", flush=True)
    print(f"[{now_ts()}] test_tsv:  {config.test_tsv}", flush=True)
    print(
        f"[{now_ts()}] seed={config.seed} epochs={config.epochs} batch_size={config.batch_size} "
        f"lr={config.lr} RMSprop loss=mse (no class_weight)",
        flush=True,
    )

    train_rows, train_seqs, y_train, _train_cols = load_tsv_rows(Path(config.train_tsv))
    val_rows, val_seqs, y_val, _val_cols = load_tsv_rows(Path(config.val_tsv))
    test_rows, test_seqs, y_test, test_cols = load_tsv_rows(Path(config.test_tsv))

    train_seqs, y_train = take_first(train_seqs, y_train, config.max_train)
    val_seqs, y_val = take_first(val_seqs, y_val, config.max_val)
    test_seqs, y_test = take_first(test_seqs, y_test, config.max_test)
    if config.max_train and config.max_train > 0:
        train_rows = train_rows[: config.max_train]
    if config.max_val and config.max_val > 0:
        val_rows = val_rows[: config.max_val]
    if config.max_test and config.max_test > 0:
        test_rows = test_rows[: config.max_test]

    print(f"[{now_ts()}] train: N={len(y_train)}, pos={int(np.sum(y_train==1))}, neg={int(np.sum(y_train==0))}, pos_ratio={(float(np.mean(y_train)) if len(y_train) else 0.0):.4f}", flush=True)
    print(f"[{now_ts()}] val: N={len(y_val)}, pos={int(np.sum(y_val==1))}, neg={int(np.sum(y_val==0))}, pos_ratio={(float(np.mean(y_val)) if len(y_val) else 0.0):.4f}", flush=True)
    print(f"[{now_ts()}] test: N={len(y_test)}, pos={int(np.sum(y_test==1))}, neg={int(np.sum(y_test==0))}, pos_ratio={(float(np.mean(y_test)) if len(y_test) else 0.0):.4f}", flush=True)

    save_json(
        paths.dataset_stats_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "train": split_stats("train", y_train),
            "val": split_stats("val", y_val),
            "test": split_stats("test", y_test),
        },
    )

    vocab = build_vocab()
    preprocess = {
        "seq_len": config.seq_len,
        "vocab": vocab,
        "cleaning": {
            "padding_token": "_",
            "unknown_token": "X",
            "keep_existing_x": True,
            "map_non_standard_to": "X",
            "pad_truncate_to": config.seq_len,
        },
        "train_tsv": config.train_tsv,
        "val_tsv": config.val_tsv,
        "test_tsv": config.test_tsv,
    }
    save_json(paths.preprocess_path, preprocess)

    x_train = encode_sequences(
        train_seqs, vocab=vocab, fixed_len=config.seq_len, split_label="train"
    )
    x_val = encode_sequences(val_seqs, vocab=vocab, fixed_len=config.seq_len, split_label="val")
    x_test = encode_sequences(test_seqs, vocab=vocab, fixed_len=config.seq_len, split_label="test")

    optimizer = build_rmsprop(learning_rate=float(config.lr))
    model = build_model(
        vocab_size=max(vocab.values()) + 1,
        seq_len=config.seq_len,
        embedding_dim=config.embedding_dim,
        conv_filters=config.conv_filters,
        gru_units=config.gru_units,
        dropout=config.dropout,
        optimizer=optimizer,
    )

    summary_lines: List[str] = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    (paths.model_extra_dir / "model_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines), flush=True)

    from tensorflow import keras

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(paths.best_model_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.patience,
        restore_best_weights=False,
        verbose=1,
    )
    val_extra = ValMetricsCallback(
        x_val=x_val,
        y_val=y_val,
        batch_size=config.batch_size,
        every_n_epochs=1,
    ).to_keras_callback()

    print(f"[{now_ts()}] Start training ...", flush=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=[checkpoint, early_stop, val_extra],
        shuffle=True,
    )

    model.save(str(paths.final_model_path))
    pd.DataFrame(history.history).to_csv(paths.training_log_path, index=False)
    print(f"[{now_ts()}] Training done. Saved last model: {paths.final_model_path}", flush=True)
    print(f"[{now_ts()}] Best model (by val_auc / AUROC): {paths.best_model_path}", flush=True)

    from model import load_model

    print(f"[{now_ts()}] Loading best model for independent testing ...", flush=True)
    best_model = load_model(str(paths.best_model_path))
    best_epoch = None
    try:
        best_epoch = int(np.nanargmax(np.asarray(history.history.get("val_auc", []), dtype=float)) + 1)
    except Exception:
        best_epoch = None

    decision_cutoff = float(config.decision_threshold)
    y_val_prob = best_model.predict(x_val, batch_size=config.batch_size, verbose=0).reshape(-1)
    metrics_val = compute_metrics(y_val, y_val_prob, threshold=decision_cutoff)

    y_test_prob = best_model.predict(x_test, batch_size=config.batch_size, verbose=0).reshape(-1)
    if int(y_test_prob.shape[0]) != int(len(test_rows)):
        raise RuntimeError(
            f"Prediction count mismatch: pred={int(y_test_prob.shape[0])} vs test_rows={int(len(test_rows))}. Refuse to write results."
        )
    if int(len(test_seqs)) != int(len(test_rows)):
        raise RuntimeError(
            f"Sequence count mismatch: test_seqs={int(len(test_seqs))} vs test_rows={int(len(test_rows))}. Refuse to write results."
        )

    metrics_test = compute_metrics(y_test, y_test_prob, threshold=decision_cutoff)

    extra_cols = [column for column in test_cols if column not in {"Sequence", "Label"}]
    pred_df = build_prediction_dataframe(test_rows, extra_cols, y_test_prob, threshold=decision_cutoff)
    save_predictions(paths.test_predictions_path, pred_df)

    val_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        n=len(y_val),
        metrics_dict=metrics_val,
        primary_result="best",
        best_epoch=best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.keras",
    )
    test_payload = build_standard_metrics_payload(
        method=METHOD_NAME,
        n=len(y_test),
        metrics_dict=metrics_test,
        primary_result="best",
        best_epoch=best_epoch,
        predictions_relpath="results/test_predictions.tsv",
        best_model_relpath="model/best_model.keras",
    )
    save_json(paths.val_metrics_path, val_payload)
    save_json(paths.test_metrics_path, test_payload)
    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "config": config.to_dict(),
            "best_epoch": best_epoch,
        },
    )

    print(f"[{now_ts()}] Independent TEST metrics:", flush=True)
    print(
        f"[{now_ts()}] "
        f"AUC={metrics_test['auc']:.4f} AUPRC={metrics_test['auprc']:.4f} "
        f"ACC={metrics_test['acc']:.4f} MCC={metrics_test['mcc']:.4f} "
        f"F1={metrics_test['f1']:.4f} Precision={metrics_test['precision']:.4f} Recall(Sn)={metrics_test['sn']:.4f} Sp={metrics_test['sp']:.4f}",
        flush=True,
    )
    print(
        f"[{now_ts()}] Confusion: tp={int(metrics_test['tp'])} tn={int(metrics_test['tn'])} "
        f"fp={int(metrics_test['fp'])} fn={int(metrics_test['fn'])}",
        flush=True,
    )
    print(
        f"[{now_ts()}] Saved: {output_dir}/ (model/best_model.keras, model/final_model.keras, preprocess.json, results/test_predictions.tsv, results/test_metrics.json)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
