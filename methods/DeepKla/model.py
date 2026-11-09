from __future__ import annotations

import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

_ATTENTION_WITH_CONTEXT_CLS = None


def configure_tensorflow_memory_growth() -> None:
    """Avoid TF grabbing nearly all VRAM at process start; must run before GPU ops."""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[DeepKla] TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s).", flush=True)
    except Exception as exc:
        print(f"[DeepKla] WARN: failed to configure TensorFlow memory growth: {exc}", flush=True)


def build_rmsprop(learning_rate: float):
    import tensorflow as tf

    return tf.keras.optimizers.RMSprop(learning_rate=float(learning_rate))


def get_attention_layer_class():
    global _ATTENTION_WITH_CONTEXT_CLS
    if _ATTENTION_WITH_CONTEXT_CLS is not None:
        return _ATTENTION_WITH_CONTEXT_CLS

    import tensorflow as tf
    from tensorflow import keras

    @keras.utils.register_keras_serializable(package="DeepKla")
    class AttentionWithContext(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            feat_dim = int(input_shape[-1])
            self.W = self.add_weight(
                name="att_W",
                shape=(feat_dim, feat_dim),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b = self.add_weight(
                name="att_b",
                shape=(feat_dim,),
                initializer="zeros",
                trainable=True,
            )
            self.u = self.add_weight(
                name="att_u",
                shape=(feat_dim,),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            ait = tf.tensordot(uit, self.u, axes=1)
            a = tf.nn.softmax(ait, axis=1)
            a = tf.expand_dims(a, axis=-1)
            weighted = x * a
            return tf.reduce_sum(weighted, axis=1)

        def get_config(self):
            return super().get_config()

    _ATTENTION_WITH_CONTEXT_CLS = AttentionWithContext
    return _ATTENTION_WITH_CONTEXT_CLS


def build_model(
    *,
    vocab_size: int,
    seq_len: int,
    embedding_dim: int,
    conv_filters: int,
    gru_units: int,
    dropout: float,
    optimizer,
):
    from tensorflow import keras

    AttentionWithContext = get_attention_layer_class()

    inp = keras.Input(shape=(seq_len,), dtype="int32", name="seq")
    x = keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_len, name="emb")(inp)
    x = keras.layers.Dropout(dropout, name="emb_dropout")(x)
    x = keras.layers.Conv1D(filters=conv_filters, kernel_size=10, activation="relu", padding="valid", name="conv1")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = keras.layers.Conv1D(filters=conv_filters, kernel_size=5, activation="relu", padding="valid", name="conv2")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(gru_units, return_sequences=True),
        name="bigru",
    )(x)
    x = AttentionWithContext(name="att")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=inp, outputs=out, name=f"DeepKla{int(seq_len)}_CNN_BiGRU_Att")
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="auprc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def load_model(model_path: str):
    from tensorflow import keras

    AttentionWithContext = get_attention_layer_class()
    return keras.models.load_model(
        str(model_path),
        custom_objects={"AttentionWithContext": AttentionWithContext},
    )
