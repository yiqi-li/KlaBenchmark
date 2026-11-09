from __future__ import annotations

import atexit
import io
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
from dataset import (
    build_dataset_stats,
    encode_sequences,
    load_benchmark_triplet,
    print_dataset_summary,
    proteinbert_compatible_seqs,
    proteinbert_token_len,
    validate_prediction_scores,
)
from model import (
    build_finetuning_components,
    build_inference_model,
    configure_tensorflow_environment,
    configure_tensorflow_memory_growth,
    import_tensorflow_or_raise,
    save_encoder_artifacts,
)
from utils import (
    TeeRunLog,
    build_prediction_dataframe,
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    format_float,
    save_curves,
    save_json,
    save_predictions,
    set_seed,
)


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    run_name = config.run_name or default_run_name(config.train_tsv)
    config.run_name = run_name
    return (Path(config.out_root) / run_name).resolve()


def train_stage(
    *,
    stage_name: str,
    model_generator,
    input_encoder,
    output_spec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seq_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    freeze_pretrained_layers: bool,
    checkpoint_dir: Path,
    training_log_rows: list | None = None,
):
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_curve,
        roc_auc_score,
    )
    from tensorflow import keras

    from proteinbert.finetuning import encode_dataset
    from proteinbert.model_generation import _slice_arrays

    raw_seq_len = int(seq_len)
    token_seq_len = proteinbert_token_len(raw_seq_len)

    train_seqs = proteinbert_compatible_seqs(
        train_df["Sequence"],
        raw_seq_len,
        split_label=f"{stage_name} train",
    )
    val_seqs = proteinbert_compatible_seqs(
        val_df["Sequence"],
        raw_seq_len,
        split_label=f"{stage_name} val",
    )

    train_X, train_y, train_sw = encode_dataset(
        train_seqs,
        train_df["Label"],
        input_encoder,
        output_spec,
        seq_len=token_seq_len,
        needs_filtering=False,
        dataset_name=f"{stage_name} train",
        verbose=True,
    )
    val_X, val_y, val_sw = encode_dataset(
        val_seqs,
        val_df["Label"],
        input_encoder,
        output_spec,
        seq_len=token_seq_len,
        needs_filtering=False,
        dataset_name=f"{stage_name} val",
        verbose=True,
    )

    y_val_true = np.asarray(val_y).reshape(-1).astype(int)
    ensure_dir(checkpoint_dir)
    best_weights_path = checkpoint_dir / f"best_{stage_name}_seq{seq_len}.weights.h5"

    class ValMetricsCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_auroc = -1.0
            self.best_epoch = -1

        def on_train_begin(self, logs=None):
            print(
                f"[PBertKla][{stage_name}] start: "
                f"raw_seq_len={raw_seq_len} proteinbert_token_len={token_seq_len} "
                f"batch_size={batch_size} lr={lr} "
                f"freeze_pretrained_layers={freeze_pretrained_layers} epochs={n_epochs}"
            )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            y_score = self.model.predict(val_X, batch_size=batch_size, verbose=0).reshape(-1)

            try:
                auroc = float(roc_auc_score(y_val_true, y_score))
            except Exception:
                auroc = float("nan")

            precision, recall, _ = precision_recall_curve(y_val_true, y_score)
            auprc = float(auc(recall, precision))

            y_pred = (y_score >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val_true, y_pred, labels=[0, 1]).ravel()
            acc = float(accuracy_score(y_val_true, y_pred))
            sn = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            sp = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
            mcc = float(matthews_corrcoef(y_val_true, y_pred))

            loss = float(logs.get("loss", float("nan")))
            val_loss = float(logs.get("val_loss", float("nan")))
            epoch_display = epoch + 1

            try:
                logs["val_AUROC"] = auroc
                logs["val_AUPRC"] = auprc
                logs["val_Acc"] = acc
                logs["val_Sn"] = sn
                logs["val_Sp"] = sp
                logs["val_MCC"] = mcc
            except Exception:
                pass

            print(
                f"[PBertKla][{stage_name}] epoch {epoch_display}/{n_epochs} "
                f"loss={format_float(loss)} val_loss={format_float(val_loss)} "
                f"val_AUROC={format_float(auroc)} val_AUPRC={format_float(auprc)} "
                f"val_Acc={format_float(acc)} val_Sn={format_float(sn)} "
                f"val_Sp={format_float(sp)} val_MCC={format_float(mcc)}"
            )
            if training_log_rows is not None:
                training_log_rows.append(
                    {
                        "stage": stage_name,
                        "epoch": int(epoch_display),
                        "train_loss": float(loss),
                        "val_loss": float(val_loss),
                        "val_auroc": float(auroc),
                        "val_auprc": float(auprc),
                        "val_accuracy": float(acc),
                        "val_recall": float(sn),
                        "val_specificity": float(sp),
                        "val_mcc": float(mcc),
                    }
                )

            if not np.isnan(auroc) and auroc > self.best_auroc + 1e-6:
                self.best_auroc = auroc
                self.best_epoch = epoch_display
                self.model.save_weights(best_weights_path)
                print(
                    f"[PBertKla][{stage_name}] BEST updated: "
                    f"epoch={self.best_epoch} val_AUROC={format_float(self.best_auroc)} "
                    f"-> saved {best_weights_path}"
                )

    callbacks = [
        ValMetricsCallback(),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_AUROC",
            mode="max",
            patience=1,
            factor=0.25,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_AUROC",
            mode="max",
        ),
    ]

    model_generator.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_y, slice(0, 1)))
    model = model_generator.create_model(token_seq_len, freeze_pretrained_layers=freeze_pretrained_layers)
    if lr is not None:
        model.optimizer.lr = lr
    model.fit(
        train_X,
        train_y,
        sample_weight=train_sw,
        validation_data=(val_X, val_y, val_sw),
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model_generator.update_state(model)

    if not best_weights_path.exists():
        model.save_weights(best_weights_path)
        print(f"[PBertKla][{stage_name}] BEST fallback: saved last weights to {best_weights_path}")

    best_auroc = -1.0
    best_epoch = None
    for callback in callbacks:
        if hasattr(callback, "best_auroc"):
            best_auroc = float(getattr(callback, "best_auroc"))
            best_epoch = int(getattr(callback, "best_epoch"))
            break
    return best_auroc, best_weights_path, best_epoch


def train() -> int:
    parser = make_train_parser()
    args = parser.parse_args()
    config, _, _ = config_and_overrides_from_args(args)

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
    dataset_name = Path(config.train_tsv).resolve().parent.name

    paths = build_run_paths(output_dir)
    for directory in [
        paths.output_dir,
        paths.results_dir,
        paths.model_dir,
        paths.checkpoint_dir,
        paths.model_extra_dir,
        paths.cache_dir,
    ]:
        ensure_dir(directory)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    set_seed(config.seed)
    tf, output_spec, input_encoder, model_generator = build_finetuning_components(config, script_dir)
    _ = tf

    train_df, val_df, raw_test_df, test_df = load_benchmark_triplet(
        Path(config.train_tsv),
        Path(config.val_tsv),
        Path(config.test_tsv),
    )
    print_dataset_summary(train_df, val_df, test_df)

    save_json(
        paths.dataset_stats_path,
        build_dataset_stats(METHOD_NAME, dataset_name, train_df, val_df, test_df),
    )
    save_json(
        paths.run_config_path,
        {
            **config.to_dict(),
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
        },
    )

    print(f"[PBertKla] Output dir: {output_dir}")
    print(f"[PBertKla] Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(
        f"[PBertKla] Learning rates: stage1={config.lr_stage1} "
        f"stage2={config.lr_stage2} stage3={config.lr_stage3}"
    )

    best_overall = {"auroc": -1.0, "weights_path": None, "stage": None, "best_epoch": None}
    training_log_rows: list[dict] = []

    stage1_auroc, stage1_path, stage1_epoch = train_stage(
        stage_name="stage1_frozen",
        model_generator=model_generator,
        input_encoder=input_encoder,
        output_spec=output_spec,
        train_df=train_df,
        val_df=val_df,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        n_epochs=config.epochs,
        lr=config.lr_stage1,
        freeze_pretrained_layers=True,
        checkpoint_dir=paths.checkpoint_dir,
        training_log_rows=training_log_rows,
    )
    if stage1_auroc > best_overall["auroc"]:
        best_overall = {
            "auroc": stage1_auroc,
            "weights_path": str(stage1_path),
            "stage": "stage1_frozen",
            "best_epoch": stage1_epoch,
        }

    stage2_auroc, stage2_path, stage2_epoch = train_stage(
        stage_name="stage2_unfrozen",
        model_generator=model_generator,
        input_encoder=input_encoder,
        output_spec=output_spec,
        train_df=train_df,
        val_df=val_df,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        n_epochs=config.epochs,
        lr=config.lr_stage2,
        freeze_pretrained_layers=False,
        checkpoint_dir=paths.checkpoint_dir,
        training_log_rows=training_log_rows,
    )
    if stage2_auroc > best_overall["auroc"]:
        best_overall = {
            "auroc": stage2_auroc,
            "weights_path": str(stage2_path),
            "stage": "stage2_unfrozen",
            "best_epoch": stage2_epoch,
        }

    if config.n_final_epochs and config.n_final_epochs > 0:
        final_batch_size = max(int(config.batch_size / (config.final_seq_len / config.seq_len)), 1)
        stage3_auroc, stage3_path, stage3_epoch = train_stage(
            stage_name="stage3_final",
            model_generator=model_generator,
            input_encoder=input_encoder,
            output_spec=output_spec,
            train_df=train_df,
            val_df=val_df,
            seq_len=config.final_seq_len,
            batch_size=final_batch_size,
            n_epochs=config.n_final_epochs,
            lr=config.lr_stage3,
            freeze_pretrained_layers=False,
            checkpoint_dir=paths.checkpoint_dir,
            training_log_rows=training_log_rows,
        )
        if stage3_auroc > best_overall["auroc"]:
            best_overall = {
                "auroc": stage3_auroc,
                "weights_path": str(stage3_path),
                "stage": "stage3_final",
                "best_epoch": stage3_epoch,
            }

    print(
        f"[PBertKla] Best checkpoint overall: stage={best_overall['stage']} "
        f"val_AUROC={format_float(best_overall['auroc'])} path={best_overall['weights_path']}"
    )
    pd.DataFrame(training_log_rows).to_csv(paths.training_log_path, index=False)

    print("[PBertKla] Saving model artifacts...")
    model = build_inference_model(model_generator, config.seq_len, best_overall["weights_path"])
    if best_overall.get("weights_path") is not None:
        print(f"[PBertKla] Loaded best checkpoint weights for saving/testing: {best_overall['weights_path']}")

    try:
        model.save(paths.model_extra_dir / "saved_model", include_optimizer=False)
    except Exception as exc:
        print(f"[PBertKla] WARN: model.save(SavedModel) failed: {exc}")

    model.save_weights(paths.best_model_weights_path)
    try:
        (paths.model_extra_dir / "model_architecture.json").write_text(model.to_json(), encoding="utf-8")
    except Exception as exc:
        print(f"[PBertKla] WARN: model.to_json() failed (will save model_summary.txt instead): {exc}")
        buffer = io.StringIO()
        model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        (paths.model_extra_dir / "model_summary.txt").write_text(buffer.getvalue(), encoding="utf-8")

    save_json(
        paths.preprocess_path,
        {
            "cleaning": {
                "raw_sequence_semantics": {"_": "padding", "X": "unknown"},
                "proteinbert_input_conversion": "locally map '_' and other non-standard chars to 'X' before ProteinBERT encoding",
            },
            "expected_columns": ["Sequence", "Label"],
            "label_values": [0, 1],
            "seq_len": int(config.seq_len),
            "seq_len_semantics": "raw_amino_acid_length_excluding_START_END",
            "proteinbert_token_len": proteinbert_token_len(int(config.seq_len)),
        },
    )
    save_encoder_artifacts(input_encoder, output_spec, paths.input_encoder_path, paths.output_spec_path)

    X_val_eval, _ = encode_sequences(
        input_encoder,
        val_df["Sequence"],
        config.seq_len,
        split_label="val",
    )
    y_val_score = model.predict(X_val_eval, batch_size=config.batch_size).reshape(-1)
    val_metrics = compute_metrics(
        val_df["Label"].to_numpy().astype(int),
        y_val_score,
        threshold=float(config.decision_threshold),
    )

    print("[PBertKla] Evaluating on test set...")
    X_test, _ = encode_sequences(
        input_encoder,
        test_df["Sequence"],
        config.seq_len,
        split_label="test",
    )
    y_score = model.predict(X_test, batch_size=config.batch_size).reshape(-1)
    validate_prediction_scores(y_score, len(test_df))
    y_true = test_df["Label"].to_numpy().astype(int)
    test_metrics = compute_metrics(y_true, y_score, threshold=float(config.decision_threshold))

    save_json(
        paths.test_metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=test_metrics,
            primary_result="best",
            best_epoch=best_overall["best_epoch"],
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath="model/best_model.weights.h5",
        ),
    )
    save_json(
        paths.val_metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=val_metrics,
            primary_result="best",
            best_epoch=best_overall["best_epoch"],
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath="model/best_model.weights.h5",
        ),
    )
    save_curves(test_metrics, paths.curves_path)

    prediction_df = build_prediction_dataframe(raw_test_df, y_score, float(config.decision_threshold))
    if int(len(prediction_df)) != int(len(test_df)):
        raise RuntimeError(f"[PBertKla][FATAL] len(pred_df)={len(prediction_df)} != len(test_df)={len(test_df)}")
    save_predictions(paths.test_predictions_path, prediction_df)

    save_json(
        paths.run_meta_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "config": config.to_dict(),
            "best_overall": best_overall,
        },
    )

    print("[PBertKla] Done.")
    print(
        f"[PBertKla] Test metrics: "
        f"AUROC={test_metrics['AUROC']:.4f} AUPRC={test_metrics['AUPRC']:.4f} "
        f"Acc={test_metrics['Accuracy']:.4f} Sn={test_metrics['Sensitivity']:.4f} "
        f"Sp={test_metrics['Specificity']:.4f} MCC={test_metrics['MCC']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
