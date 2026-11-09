from __future__ import annotations

import atexit
import os
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from configs import METHOD_NAME, config_and_overrides_from_args, make_train_parser
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
    build_run_paths,
    build_standard_metrics_payload,
    compute_metrics,
    default_run_name,
    ensure_dir,
    pick_device,
    save_json,
    save_predictions,
    set_seed,
    split_stats,
    validate_prediction_alignment,
)


def resolve_output_dir(config) -> Path:
    if config.output_dir:
        return Path(config.output_dir).resolve()
    run_name = config.run_name or default_run_name(config.train_tsv)
    config.run_name = run_name
    return (Path(config.out_root) / run_name).resolve()


def evaluate(
    classifier: PCBertKlaClassifier,
    bert,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return average loss and aligned probabilities in dataset order."""
    classifier.eval()
    bert.eval()

    num_samples = len(loader.dataset)  # type: ignore[arg-type]
    y_true = np.empty((num_samples,), dtype=np.int64)
    y_prob = np.empty((num_samples,), dtype=np.float32)
    seen = np.zeros((num_samples,), dtype=bool)

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, phys, labels, indices in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            phys = phys.to(device)
            labels = labels.to(device)

            embeddings = bert(**inputs).last_hidden_state[:, 0, :]
            fused = torch.cat([embeddings, phys], dim=1)
            logits = classifier(fused)
            probabilities = torch.sigmoid(logits)
            loss = criterion(probabilities, labels)

            batch_size = int(logits.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            probability_np = probabilities.detach().cpu().numpy().astype(np.float32)
            index_np = indices.detach().cpu().numpy().astype(np.int64)

            if index_np.ndim != 1:
                raise RuntimeError(f"Unexpected idx shape: {index_np.shape}")
            if (index_np < 0).any() or (index_np >= num_samples).any():
                raise RuntimeError(
                    f"Index out of range: min={index_np.min()} max={index_np.max()} n={num_samples}"
                )
            if seen[index_np].any():
                duplicate = index_np[seen[index_np]][0]
                raise RuntimeError(f"Detected duplicated idx={int(duplicate)} during evaluation.")

            seen[index_np] = True
            y_true[index_np] = labels.detach().cpu().numpy().astype(np.int64)
            y_prob[index_np] = probability_np

    if not seen.all():
        missing = np.where(~seen)[0]
        raise RuntimeError(f"Prediction alignment failed. Missing sample indices: {missing[:10].tolist()}")

    average_loss = float(total_loss / max(1, total_count))
    return average_loss, y_true, y_prob


def predict_scores(
    classifier: PCBertKlaClassifier,
    bert,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    _, y_true, y_prob = evaluate(classifier, bert, loader, device, criterion)
    return y_true, y_prob


def train() -> None:
    parser = make_train_parser()
    args = parser.parse_args()
    config, _, _ = config_and_overrides_from_args(args)

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print(f"[{METHOD_NAME}] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    set_seed(config.seed)
    device = pick_device(config.device)

    output_dir = resolve_output_dir(config)
    config.output_dir = str(output_dir)

    dataset_name = Path(config.train_tsv).resolve().parent.name
    paths = build_run_paths(output_dir)
    for directory in [output_dir, paths.model_dir, paths.model_extra_dir, paths.results_dir, paths.cache_dir]:
        ensure_dir(directory)

    run_log = TeeRunLog(output_dir)
    run_log.start()
    atexit.register(run_log.stop)

    print("Starting end-to-end training and evaluation...")
    print(
        f"Device: {device} | keep_bert_layers={config.keep_bert_layers} | "
        f"lr_bert={config.lr_bert} | lr_other={config.lr_other} | weight_decay={config.weight_decay}"
    )

    save_json(
        paths.run_config_path,
        {
            **config.to_dict(),
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
        },
    )

    train_df = load_tsv(Path(config.train_tsv))
    val_df = load_tsv(Path(config.val_tsv))
    test_df = load_tsv(Path(config.test_tsv))

    x_train_raw, y_train, train_seq_col = extract_sequences_and_labels(train_df, require_label=True)
    x_val_raw, y_val, val_seq_col = extract_sequences_and_labels(val_df, require_label=True)
    x_test_raw, y_test, test_seq_col = extract_sequences_and_labels(test_df, require_label=True)

    assert y_train is not None
    assert y_val is not None
    assert y_test is not None

    save_json(
        paths.preprocess_path,
        {
            "method": METHOD_NAME,
            "dataset_name": dataset_name,
            "seq_len": int(config.seq_len),
            "train_seq_col": train_seq_col,
            "val_seq_col": val_seq_col,
            "test_seq_col": test_seq_col,
            "physchem_dim": 27,
            "mode": "finetune_protbert_official_like",
        },
    )

    if config.limit_train > 0:
        x_train_raw, y_train = x_train_raw[: config.limit_train], y_train[: config.limit_train]
    if config.limit_val > 0:
        x_val_raw, y_val = x_val_raw[: config.limit_val], y_val[: config.limit_val]
    if config.limit_test > 0:
        x_test_raw, y_test = x_test_raw[: config.limit_test], y_test[: config.limit_test]

    print(f"Train samples: {len(x_train_raw)} | Val samples: {len(x_val_raw)} | Test samples: {len(x_test_raw)}")
    print(f"Train file: {config.train_tsv}")
    print(f"Val file:   {config.val_tsv}")
    print(f"Test file:  {config.test_tsv}\n")

    warn_raw_seq_len_adjustments(x_train_raw, int(config.seq_len), split_label="train")
    warn_raw_seq_len_adjustments(x_val_raw, int(config.seq_len), split_label="val")
    warn_raw_seq_len_adjustments(x_test_raw, int(config.seq_len), split_label="test")

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

    phys_train = compute_physchem_features(x_train_raw, seq_len=int(config.seq_len))
    phys_val = compute_physchem_features(x_val_raw, seq_len=int(config.seq_len))
    phys_test = compute_physchem_features(x_test_raw, seq_len=int(config.seq_len))

    scaler = MinMaxScaler()
    scaler.fit(phys_train)
    joblib.dump(scaler, paths.scaler_path)

    phys_train_scaled = scaler.transform(phys_train).astype(np.float32)
    phys_val_scaled = scaler.transform(phys_val).astype(np.float32)
    phys_test_scaled = scaler.transform(phys_test).astype(np.float32)

    tokenizer, bert, max_length_tokens = build_tokenizer_and_backbone(config, paths.cache_dir, device)
    collate_fn = build_collate_fn(tokenizer, seq_len=int(config.seq_len), max_length_tokens=max_length_tokens)

    train_loader = DataLoader(
        SequencePhysDataset(x_train_raw, phys_train_scaled, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SequencePhysDataset(x_val_raw, phys_val_scaled, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SequencePhysDataset(x_test_raw, phys_test_scaled, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    classifier = PCBertKlaClassifier(input_dim=CLASSIFIER_INPUT_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        [
            {"params": classifier.parameters(), "lr": float(config.lr_other)},
            {"params": bert.parameters(), "lr": float(config.lr_bert)},
        ],
        weight_decay=config.weight_decay,
    )

    best_val_auroc = -1.0
    best_classifier_state = None
    best_bert_state = None
    patience_left = config.patience
    best_epoch = None
    epoch_rows = []

    for epoch in range(1, config.epochs + 1):
        classifier.train()
        bert.train()
        total_loss = 0.0

        for inputs, phys, labels, _ in train_loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            phys = phys.to(device)
            labels = labels.to(device)

            embeddings = bert(**inputs).last_hidden_state[:, 0, :]
            fused = torch.cat([embeddings, phys], dim=1)
            logits = classifier(fused)
            probabilities = torch.sigmoid(logits)
            loss = criterion(probabilities, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_loss, y_val_true, y_val_prob = evaluate(classifier, bert, val_loader, device, criterion)
        val_metrics = compute_metrics(y_val_true, y_val_prob, threshold=0.5)
        train_loss = total_loss / max(1, len(train_loader))

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_pre={val_metrics['precision']:.4f} "
            f"val_rec={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_mcc={val_metrics['mcc']:.4f} "
            f"val_sp={val_metrics['specificity']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f}"
        )
        epoch_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auroc": float(val_metrics["auc"]),
                "val_auprc": float(val_metrics["auprc"]),
                "val_accuracy": float(val_metrics["acc"]),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_specificity": float(val_metrics["specificity"]),
                "val_f1": float(val_metrics["f1"]),
                "val_mcc": float(val_metrics["mcc"]),
            }
        )

        current_auc = float(val_metrics["auc"]) if not np.isnan(val_metrics["auc"]) else -float("inf")
        if current_auc > best_val_auroc:
            best_val_auroc = current_auc
            best_epoch = epoch
            print(f"New best val_auroc={best_val_auroc:.4f} at epoch {epoch}.")
            best_classifier_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            best_bert_state = {key: value.detach().cpu().clone() for key, value in bert.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping triggered (best_val_auroc={best_val_auroc:.4f}).")
                break

    if best_classifier_state is None:
        best_classifier_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    if best_bert_state is None:
        best_bert_state = {key: value.detach().cpu().clone() for key, value in bert.state_dict().items()}

    classifier.load_state_dict(best_classifier_state)
    bert.load_state_dict(best_bert_state)

    pd.DataFrame(epoch_rows).to_csv(paths.training_log_path, index=False)

    torch.save(
        {
            "classifier_state_dict": best_classifier_state,
            "protbert_state_dict": best_bert_state,
            "input_dim": CLASSIFIER_INPUT_DIM,
            "keep_bert_layers": int(config.keep_bert_layers),
        },
        paths.model_path,
    )

    decision_cutoff = float(config.decision_threshold)
    y_val_true_final, y_val_prob_final = predict_scores(classifier, bert, val_loader, device, criterion)
    val_metrics = compute_metrics(y_val_true_final, y_val_prob_final, threshold=decision_cutoff)

    y_test_true, y_test_prob = predict_scores(classifier, bert, test_loader, device, criterion)
    test_metrics = compute_metrics(y_test_true, y_test_prob, threshold=decision_cutoff)

    print("==== Independent test metrics ====")
    for metric_name in ["acc", "precision", "recall", "f1", "mcc", "specificity", "auc", "auprc"]:
        print(f"{metric_name}: {test_metrics[metric_name]:.6f}")

    test_df_used = test_df.iloc[: len(x_test_raw)].copy()
    validate_prediction_alignment(test_df_used, test_seq_col, x_test_raw)
    prediction_frame = build_prediction_dataframe(
        test_df_used,
        seq_col=test_seq_col,
        probabilities=y_test_prob,
        threshold=decision_cutoff,
    )
    save_predictions(paths.test_predictions_path, prediction_frame)

    save_json(
        paths.val_metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=val_metrics,
            primary_result="best",
            best_epoch=best_epoch,
            predictions_relpath=None,
            best_model_relpath="model/best_model.pt",
        ),
    )
    save_json(
        paths.metrics_path,
        build_standard_metrics_payload(
            method=METHOD_NAME,
            raw_metrics=test_metrics,
            primary_result="best",
            best_epoch=best_epoch,
            predictions_relpath="results/test_predictions.tsv",
            best_model_relpath="model/best_model.pt",
        ),
    )

    save_json(
        paths.run_meta_path,
        {
            "paths": {key: str(value) for key, value in asdict(paths).items()},
            "config": config.to_dict(),
            "device": str(device),
            "train_seq_col": train_seq_col,
            "val_seq_col": val_seq_col,
            "test_seq_col": test_seq_col,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "optim": {
                "name": "AdamW",
                "lr_other": float(config.lr_other),
                "lr_bert": float(config.lr_bert),
                "weight_decay": float(config.weight_decay),
            },
            "protbert": {
                "mode": "finetune_protbert_official_like",
                "model_name": config.protbert_dir,
                "keep_bert_layers": int(config.keep_bert_layers),
                "local_files_only": bool(config.local_files_only),
            },
        },
    )

    if config.save_pretrained:
        save_dir = paths.model_extra_dir / "protbert_finetuned"
        ensure_dir(save_dir)
        bert.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    print(
        "\nSaved:\n"
        f"- Model: {paths.model_path}\n"
        f"- Scaler: {paths.scaler_path}\n"
        f"- Test metrics: {paths.metrics_path}\n"
        f"- Test predictions: {paths.test_predictions_path}\n"
        f"- Run metadata: {paths.run_meta_path}"
    )


if __name__ == "__main__":
    train()
