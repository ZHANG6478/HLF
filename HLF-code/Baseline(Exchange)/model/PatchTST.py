import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler


def _make_supervised(series_1d_scaled: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(series_1d_scaled)):
        win = series_1d_scaled[i - seq_len:i]
        target = series_1d_scaled[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Cannot create valid sequences (seq_len={seq_len})")
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

    def call(self, x):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        C = tf.shape(x)[2]

        p = tf.minimum(self.patch_len, L)
        L_trim = (L // p) * p
        x = x[:, L - L_trim:, :]
        T = L_trim // p

        x = tf.reshape(x, [B, T, p, C])
        x = tf.reshape(x, [B, T, p * C])
        return x


class TransformerEncoder(layers.Layer):
    def __init__(self, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        attn = self.mha(x, x, training=training)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)

        f = self.ffn(x, training=training)
        f = self.drop2(f, training=training)
        x = self.norm2(x + f)
        return x


def _build_patchtst(seq_len, patch_len=2, d_model=32, n_heads=2, d_ff=64, n_layers=1, dropout=0.2):
    inp = layers.Input(shape=(seq_len, 1), name="input_window")

    x = Patchify(patch_len)(inp)
    x = layers.Dense(d_model)(x)
    x = PositionEmbedding(max_len=max(1, int(seq_len)), d_model=d_model)(x)
    x = layers.Dropout(dropout)(x)

    for _ in range(int(n_layers)):
        x = TransformerEncoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)(x)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)

    return tf.keras.Model(inp, out, name="PatchTST_OneStep")


def patchtst_predict(
    train_data,
    test_data,
    seq_len=7,
    epochs=60,
    batch_size=32,
    learning_rate=1e-3,
    refit_interval=7,
    patch_len=1,
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=2,
    dropout=0.1,
):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= seq_len:
        raise ValueError(f"train_data length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    patch_len = int(max(1, min(patch_len, seq_len)))

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    def _fit_model(current_history_raw: np.ndarray):
        hist_scaled = scaler.transform(current_history_raw.reshape(-1, 1)).flatten()
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
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="mse")

        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=5, min_lr=1e-5)

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es, rlp],
            verbose=0,
        )
        return model

    history = train_data.tolist()
    model = _fit_model(np.asarray(history, dtype=float))

    preds = []
    for i in range(len(test_data)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        y_hat_scaled = float(model.predict(win_scaled, verbose=0).flatten()[0])
        y_hat = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        preds.append(y_hat)
        history.append(float(test_data[i]))

    return np.asarray(preds, dtype=float), test_data
