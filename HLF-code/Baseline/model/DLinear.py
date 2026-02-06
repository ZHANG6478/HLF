import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler


class RevIN(layers.Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std = None

    def call(self, x, mode="norm"):
        if mode == "norm":
            self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
            self.std = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
            return (x - self.mean) / self.std
        if mode == "denorm":
            return x * self.std + self.mean
        raise ValueError("mode must be 'norm' or 'denorm'")


def moving_avg(x, kernel_size):
    if kernel_size <= 1:
        return x
    return tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding="same")(x)


class DLinearPaper(tf.keras.Model):
    def __init__(self, seq_len, pred_len=1, ma_kernel=7, use_revin=True, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ma_kernel = ma_kernel
        self.use_revin = use_revin
        self.revin = RevIN() if use_revin else None

        self.linear_seasonal = layers.Dense(pred_len, use_bias=True)
        self.linear_trend = layers.Dense(pred_len, use_bias=True)
        self.dropout = layers.Dropout(dropout) if dropout and dropout > 0 else None

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


def _create_seq(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        win = series[i - seq_len:i]
        target = series[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Cannot create valid sequences (seq_len={seq_len})")
    return np.array(X, dtype=float), np.array(y, dtype=float)


def dlinear_predict(
    train_data,
    test_data,
    seq_len=7,
    ma_kernel=None,
    epochs=120,
    batch_size=32,
    learning_rate=1e-3,
    use_revin=True,
    dropout=0.2,
    refit_interval=7,
):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= seq_len:
        raise ValueError(f"train_data length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if ma_kernel is None:
        ma_kernel = min(25, seq_len)
    ma_kernel = max(1, int(ma_kernel))

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    def _fit_model(current_history_raw: np.ndarray):
        hist_scaled = scaler.transform(current_history_raw.reshape(-1, 1)).flatten()

        X_train, y_train = _create_seq(hist_scaled, seq_len)
        X_train = X_train.reshape(-1, seq_len, 1)
        y_train = y_train.reshape(-1, 1, 1)

        model = DLinearPaper(
            seq_len=seq_len,
            pred_len=1,
            ma_kernel=ma_kernel,
            use_revin=use_revin,
            dropout=dropout,
        )
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="mse")

        es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        rlp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=6, min_lr=1e-5)

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
    preds = []

    model = _fit_model(np.asarray(history, dtype=float))

    for i in range(len(test_data)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        y_hat_scaled = model.predict(win_scaled, verbose=0)
        y_hat_scaled = float(y_hat_scaled[0, 0, 0])
        y_hat = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        preds.append(y_hat)
        history.append(float(test_data[i]))

    return np.asarray(preds, dtype=float), test_data
