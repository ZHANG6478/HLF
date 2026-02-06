# models/patchtst.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple


def _make_supervised(series_1d_scaled: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(int(seq_len), len(series_1d_scaled)):
        win = series_1d_scaled[i - seq_len:i]
        target = series_1d_scaled[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Invalid sequence generation (seq_len={seq_len}).")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


class PositionEmbedding(layers.Layer):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.emb = layers.Embedding(input_dim=self.max_len, output_dim=self.d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        pos = tf.range(start=0, limit=T, delta=1)
        pos = self.emb(pos)
        pos = tf.expand_dims(pos, 0)
        return x + pos


class Patchify(layers.Layer):
    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.p = None
        self.out_dim = None

    def build(self, input_shape):
        L_static = input_shape[1]
        C_static = input_shape[2]
        if C_static is None:
            raise ValueError("Patchify requires a static channel dimension (e.g., 1).")

        if L_static is not None:
            self.p = int(min(self.patch_len, int(L_static)))
        else:
            self.p = int(self.patch_len)

        self.out_dim = int(self.p * int(C_static))
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        C = tf.shape(x)[2]

        p = tf.constant(self.p, dtype=tf.int32)
        L_trim = (L // p) * p

        x = x[:, L - L_trim:, :]
        T = tf.maximum(1, L_trim // p)

        def _normal():
            xx = tf.reshape(x, [B, T, p, C])
            xx = tf.reshape(xx, [B, T, self.out_dim])
            return xx

        def _fallback():
            return tf.zeros([B, 1, self.out_dim], dtype=x.dtype)

        out = tf.cond(tf.equal(L_trim, 0), _fallback, _normal)
        out = tf.ensure_shape(out, [None, None, self.out_dim])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.out_dim)


class TransformerEncoder(layers.Layer):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=int(n_heads), key_dim=int(d_model))
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(int(d_ff), activation="relu"),
                layers.Dropout(float(dropout)),
                layers.Dense(int(d_model)),
            ]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(float(dropout))
        self.drop2 = layers.Dropout(float(dropout))

    def call(self, x, training=None):
        attn = self.mha(x, x, training=training)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)

        f = self.ffn(x, training=training)
        f = self.drop2(f, training=training)
        x = self.norm2(x + f)
        return x


def _build_patchtst(
    seq_len: int,
    patch_len: int = 2,
    d_model: int = 32,
    n_heads: int = 2,
    d_ff: int = 64,
    n_layers: int = 1,
    dropout: float = 0.2,
) -> tf.keras.Model:
    seq_len = int(seq_len)
    inp = layers.Input(shape=(seq_len, 1), name="input_window")

    x = Patchify(int(patch_len))(inp)
    x = layers.Dense(int(d_model))(x)
    x = PositionEmbedding(max_len=max(1, seq_len), d_model=int(d_model))(x)
    x = layers.Dropout(float(dropout))(x)

    for _ in range(int(n_layers)):
        x = TransformerEncoder(d_model=int(d_model), n_heads=int(n_heads), d_ff=int(d_ff), dropout=float(dropout))(x)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dropout(float(dropout))(x)
    out = layers.Dense(1)(x)

    return tf.keras.Model(inp, out, name="PatchTST_OneStep")


def patchtst(
    train_series,
    test_series,
    seq_len: int = 7,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    refit_interval: Optional[int] = 7,
    patch_len: int = 1,
    d_model: int = 32,
    n_heads: int = 4,
    d_ff: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.asarray(train_series, dtype=float)
    test_data = np.asarray(test_series, dtype=float)

    seq_len = int(seq_len)
    if len(train_data) <= seq_len:
        raise ValueError(f"train_series length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    patch_len = int(max(1, min(int(patch_len), seq_len)))

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    refit_interval = int(refit_interval) if do_refit else len(test_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    def _fit_model(history_raw: np.ndarray) -> tf.keras.Model:
        hist_scaled = scaler.transform(history_raw.reshape(-1, 1)).flatten()
        X_train, y_train = _make_supervised(hist_scaled, seq_len)
        X_train = X_train.reshape(-1, seq_len, 1)

        model = _build_patchtst(
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)), loss="mse")

        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=5, min_lr=1e-5)

        model.fit(
            X_train,
            y_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=0.1,
            callbacks=[es, rlp],
            verbose=1 if verbose else 0,
        )
        return model

    history = train_data.tolist()
    model = _fit_model(np.asarray(history, dtype=float))

    preds = np.zeros(len(test_data), dtype=float)

    for i in range(len(test_data)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        y_hat_scaled = float(model.predict(win_scaled, verbose=0).flatten()[0])
        preds[i] = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        history.append(float(test_data[i]))

    return preds, test_data


patchtst_predict = patchtst
