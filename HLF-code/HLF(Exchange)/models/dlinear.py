# models/dlinear.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple


class RevIN(layers.Layer):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.mean = None
        self.std = None

    def call(self, x, mode: str = "norm"):
        if mode == "norm":
            self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
            self.std = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
            return (x - self.mean) / self.std
        if mode == "denorm":
            return x * self.std + self.mean
        raise ValueError("mode must be 'norm' or 'denorm'")


def moving_avg(x, kernel_size: int):
    if kernel_size <= 1:
        return x
    return tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding="same")(x)


class DLinearPaper(tf.keras.Model):
    def __init__(
        self,
        seq_len: int,
        pred_len: int = 1,
        ma_kernel: int = 7,
        use_revin: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.ma_kernel = int(ma_kernel)
        self.use_revin = bool(use_revin)
        self.revin = RevIN() if self.use_revin else None

        self.linear_seasonal = layers.Dense(self.pred_len, use_bias=True)
        self.linear_trend = layers.Dense(self.pred_len, use_bias=True)
        self.dropout = layers.Dropout(float(dropout)) if dropout and dropout > 0 else None

    def call(self, x, training=None):
        if self.use_revin:
            x = self.revin(x, mode="norm")

        trend = moving_avg(x, self.ma_kernel)
        seasonal = x - trend

        seasonal = tf.transpose(seasonal, perm=[0, 2, 1])
        trend = tf.transpose(trend, perm=[0, 2, 1])

        if self.dropout is not None:
            seasonal = self.dropout(seasonal, training=training)
            trend = self.dropout(trend, training=training)

        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)
        out = seasonal_out + trend_out
        out = tf.transpose(out, perm=[0, 2, 1])

        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out


def _make_supervised(series_1d: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(series_1d)):
        win = series_1d[i - seq_len:i]
        target = series_1d[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Invalid sequence generation (seq_len={seq_len}).")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def dlinear(
    train_series,
    test_series,
    seq_len: int = 7,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    ma_kernel: Optional[int] = None,
    use_revin: bool = True,
    dropout: float = 0.2,
    refit_interval: Optional[int] = 7,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_series = np.asarray(train_series, dtype=float)
    test_series = np.asarray(test_series, dtype=float)

    if len(test_series) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if len(train_series) <= int(seq_len):
        raise ValueError(f"train_series length ({len(train_series)}) must be > seq_len ({seq_len})")

    seq_len = int(seq_len)

    if ma_kernel is None:
        ma_kernel = min(25, seq_len)
    ma_kernel = max(1, int(ma_kernel))

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    refit_interval = int(refit_interval) if do_refit else len(test_series)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.reshape(-1, 1))

    def _fit_model(history_raw_1d: np.ndarray) -> tf.keras.Model:
        hist_scaled = scaler.transform(history_raw_1d.reshape(-1, 1)).flatten()
        X_train, y_train = _make_supervised(hist_scaled, seq_len)
        X_train = X_train.reshape(-1, seq_len, 1)
        y_train = y_train.reshape(-1, 1, 1)

        model = DLinearPaper(
            seq_len=seq_len,
            pred_len=1,
            ma_kernel=ma_kernel,
            use_revin=use_revin,
            dropout=dropout,
        )
        opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
        model.compile(optimizer=opt, loss="mse")

        es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=6, min_lr=1e-5)

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

    history = train_series.tolist()
    model = _fit_model(np.asarray(history, dtype=float))

    preds = np.zeros(len(test_series), dtype=float)

    for i in range(len(test_series)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        y_hat_scaled = float(model.predict(win_scaled, verbose=0)[0, 0, 0])
        preds[i] = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        history.append(float(test_series[i]))

    return preds, test_series


dlinear_predict = dlinear
