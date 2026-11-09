from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class SelfAttention(layers.Layer):
    def __init__(self, attn_size: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn_size = int(attn_size)
        self.dropout = float(dropout)
        self.supports_masking = True
        prefix = str(getattr(self, "name", "selfatt"))
        self.q = layers.Dense(self.attn_size, use_bias=False, name=f"{prefix}_q")
        self.k = layers.Dense(self.attn_size, use_bias=False, name=f"{prefix}_k")
        self.v = layers.Dense(self.attn_size, use_bias=False, name=f"{prefix}_v")
        self.proj = None

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.proj = layers.Dense(feat_dim, use_bias=False, name=f"{self.name}_proj")
        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if mask is not None:
            key_mask = tf.cast(mask, scores.dtype)
            key_mask = tf.expand_dims(key_mask, axis=1)
            scores = scores + (1.0 - key_mask) * (-1e9)

        weights = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(weights, v)
        out = self.proj(ctx)

        if mask is not None:
            query_mask = tf.cast(mask, out.dtype)
            query_mask = tf.expand_dims(query_mask, axis=-1)
            out = out * query_mask
        return out

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attn_size": self.attn_size, "dropout": self.dropout})
        return cfg


def build_abff_model(
    L: int = 31,
    vocab_size: int = 22,
    embedding_size: int = 64,
    attention_size: int = 32,
    lstm_units: int = 32,
    dense_size: int = 16,
    optimizer=None,
):
    acid_in = layers.Input(shape=(L,), dtype="int32", name="acid_input")
    cont_in = layers.Input(shape=(L,), dtype="int32", name="cont_input")

    acid_x = layers.Embedding(vocab_size, embedding_size, name="acid_embed")(acid_in)
    acid_x = SelfAttention(attention_size, dropout=0.1, name="acid_att")(acid_x)
    acid_x = layers.Dropout(0.3)(acid_x)
    acid_x = layers.Conv1D(filters=int(lstm_units / 2), kernel_size=1, activation="relu", name="acid_conv1")(acid_x)
    acid_x = layers.Dropout(0.3)(acid_x)
    acid_x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="acid_lstm",
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=regularizers.l2(0.0001),
    )(acid_x)
    acid_x = layers.Dropout(0.3)(acid_x)

    cont_x = layers.Embedding(vocab_size, embedding_size, name="cont_embed")(cont_in)
    cont_x = SelfAttention(attention_size, dropout=0.1, name="cont_att")(cont_x)
    cont_x = layers.Dropout(0.3)(cont_x)
    cont_x = layers.Conv1D(filters=int(lstm_units / 2), kernel_size=1, activation="relu", name="cont_conv1")(cont_x)
    cont_x = layers.Dropout(0.3)(cont_x)
    cont_x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="cont_lstm",
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=regularizers.l2(0.0001),
    )(cont_x)
    cont_x = layers.Dropout(0.3)(cont_x)

    merged = layers.Concatenate(name="merge_concat")([acid_x, cont_x])
    merged = SelfAttention(8, dropout=0.1, name="merge_att")(merged)
    merged = layers.Flatten()(merged)
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(
        64,
        activation="relu",
        name="merge_dense1",
        kernel_regularizer=regularizers.l2(1e-5),
        bias_regularizer=regularizers.l2(0.0001),
    )(merged)
    merged = layers.Dropout(0.3)(merged)
    merged = layers.Dense(
        dense_size,
        activation="relu",
        name="merge_dense2",
        kernel_regularizer=regularizers.l2(1e-5),
        bias_regularizer=regularizers.l2(0.0001),
    )(merged)
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(1, activation="sigmoid", name="merge_class")(merged)

    model = Model(inputs=[acid_in, cont_in], outputs=out, name="ABFF_Kla")
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(curve="ROC", name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="prauc"),
        ],
    )
    return model


ABFF_Kla_model = build_abff_model


class PrettyLogger(Callback):
    def __init__(self, train_n, val_n):
        super().__init__()
        self.train_n = train_n
        self.val_n = val_n
        self.best_auc = -np.inf

    def on_train_begin(self, logs=None):
        print("\n========== TRAINING START ==========")
        print(f"Train samples: {self.train_n}")
        print(f"Val samples:   {self.val_n}")
        print("===================================\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_auc = logs.get("val_auc")
        msg = (
            f"Epoch {epoch + 1:03d} | "
            f"loss={logs.get('loss'):.4f} | acc={logs.get('acc'):.4f} | auc={logs.get('auc'):.4f} | prauc={logs.get('prauc'):.4f} || "
            f"val_loss={logs.get('val_loss'):.4f} | val_acc={logs.get('val_acc'):.4f} | "
            f"val_auc={val_auc:.4f} | val_prauc={logs.get('val_prauc'):.4f}"
        )
        improved = (val_auc is not None) and (val_auc > self.best_auc + 1e-6)
        if improved:
            self.best_auc = val_auc
            msg += "  <-- best val_auc updated"
        print(msg)


def ensure_pretrained_models(
    run_dir: Path,
    X_train_acid: np.ndarray,
    X_train_cont: np.ndarray,
    y_train: np.ndarray,
    X_val_acid: np.ndarray,
    X_val_cont: np.ndarray,
    y_val: np.ndarray,
    *,
    L: int,
    vocab_size: int,
    embedding_size: int,
    attention_size: int,
    lstm_units: int,
    dense_size: int,
    lr: float,
    epochs: int,
    batch_size: int,
    early_patience: int,
    reduce_patience: int,
) -> tuple[Path, Path]:
    acid_ckpt = Path(run_dir) / "acid0.h5"
    contmap_ckpt = Path(run_dir) / "contmap0.h5"

    def pepti_conv_lstm_model(
        prefix: str,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        save_path: Path,
    ) -> None:
        print(f"\n========== PRETRAIN START ({prefix}) ==========")
        inp = layers.Input(shape=(L,), dtype="int32", name=f"{prefix}_input")
        x = layers.Embedding(vocab_size, embedding_size, name=f"{prefix}_embed")(inp)
        x = SelfAttention(attention_size, dropout=0.1, name=f"{prefix}_att")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv1D(filters=int(lstm_units / 2), kernel_size=1, activation="relu", name=f"{prefix}_conv1")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(
            lstm_units,
            return_sequences=False,
            name=f"{prefix}_lstm",
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.0001),
        )(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(
            dense_size,
            activation="relu",
            name=f"{prefix}_dense1",
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(0.0001),
        )(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid", name=f"{prefix}_dense2")(x)
        branch_model = Model(inputs=inp, outputs=out, name=f"{prefix}_pretrain")
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
        branch_model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(curve="ROC", name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="prauc"),
            ],
        )
        branch_model.fit(
            x=train_x,
            y=train_y,
            validation_data=(val_x, val_y),
            epochs=int(epochs),
            batch_size=int(batch_size),
            shuffle=True,
            verbose=0,
            callbacks=[
                PrettyLogger(train_n=int(len(train_y)), val_n=int(len(val_y))),
                ModelCheckpoint(
                    filepath=str(save_path),
                    monitor="val_auc",
                    mode="max",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                ),
                EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=int(early_patience),
                    restore_best_weights=True,
                    verbose=1,
                ),
            ],
        )
        branch_model.save_weights(str(save_path))
        print(f"========== PRETRAIN END ({prefix}) ==========\n")

    print(f"[pretrain] training acid branch from scratch -> {acid_ckpt}")
    pepti_conv_lstm_model("acid", X_train_acid, y_train, X_val_acid, y_val, acid_ckpt)
    print(f"[pretrain] saved: {acid_ckpt}")

    print(f"[pretrain] training contact branch from scratch -> {contmap_ckpt}")
    pepti_conv_lstm_model("cont", X_train_cont, y_train, X_val_cont, y_val, contmap_ckpt)
    print(f"[pretrain] saved: {contmap_ckpt}")
    return acid_ckpt, contmap_ckpt


def load_trained_model(path: str | Path):
    return tf.keras.models.load_model(str(path), custom_objects={"SelfAttention": SelfAttention})
