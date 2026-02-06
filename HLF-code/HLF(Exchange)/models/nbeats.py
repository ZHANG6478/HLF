import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Sequence


class NBeatsBlock(layers.Layer):
    def __init__(
        self,
        input_size: int,
        theta_size: int = 128,
        hidden_units: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = int(input_size)

        self.hidden = []
        for _ in range(int(n_hidden_layers)):
            self.hidden.append(layers.Dense(int(hidden_units), activation="relu"))
            if dropout and dropout > 0:
                self.hidden.append(layers.Dropout(float(dropout)))

        self.theta = layers.Dense(int(theta_size), activation=None)
        self.backcast = layers.Dense(self.input_size, activation=None)
        self.forecast = layers.Dense(1, activation=None)

    def call(self, x, training: bool = False):
        h = x
        for layer_ in self.hidden:
            if isinstance(layer_, layers.Dropout):
                h = layer_(h, training=training)
            else:
                h = layer_(h)

        theta = self.theta(h)
        backcast = self.backcast(theta)
        forecast = self.forecast(theta)
        return backcast, forecast


def _build_nbeats(
    input_size: int,
    n_stacks: int = 2,
    n_blocks: int = 3,
    theta_size: int = 128,
    hidden_units: int = 256,
    n_hidden_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    inp = layers.Input(shape=(int(input_size),), name="input_window")
    residual = inp
    forecast_sum = 0.0

    for _ in range(int(n_stacks)):
        for _ in range(int(n_blocks)):
            block = NBeatsBlock(
                input_size=int(input_size),
                theta_size=int(theta_size),
                hidden_units=int(hidden_units),
                n_hidden_layers=int(n_hidden_layers),
                dropout=float(dropout),
            )
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

    return tf.keras.Model(inputs=inp, outputs=forecast_sum, name="NBEATS")


def _make_supervised(series_1d_scaled: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    seq_len = int(seq_len)
    for i in range(seq_len, len(series_1d_scaled)):
        win = series_1d_scaled[i - seq_len : i]
        target = series_1d_scaled[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Cannot create valid sequences (seq_len={seq_len})")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def nbeats(
    train_series: Sequence[float],
    test_series: Sequence[float],
    seq_len: int = 7,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    refit_interval: Optional[int] = 7,
    n_stacks: int = 2,
    n_blocks: int = 3,
    theta_size: int = 128,
    hidden_units: int = 256,
    n_hidden_layers: int = 2,
    dropout: float = 0.2,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_series = np.asarray(train_series, dtype=float)
    test_series = np.asarray(test_series, dtype=float)

    if len(test_series) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if len(train_series) <= int(seq_len):
        raise ValueError(f"train_series length ({len(train_series)}) must be > seq_len ({seq_len})")

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    if not do_refit:
        refit_interval = len(test_series)
    refit_interval = int(refit_interval)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.reshape(-1, 1))

    def _fit_model(history_raw_1d: np.ndarray) -> tf.keras.Model:
        hist_scaled = scaler.transform(history_raw_1d.reshape(-1, 1)).flatten()
        X_train, y_train = _make_supervised(hist_scaled, int(seq_len))

        model = _build_nbeats(
            input_size=int(seq_len),
            n_stacks=int(n_stacks),
            n_blocks=int(n_blocks),
            theta_size=int(theta_size),
            hidden_units=int(hidden_units),
            n_hidden_layers=int(n_hidden_layers),
            dropout=float(dropout),
        )

        opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
        model.compile(optimizer=opt, loss="mse")

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

    history = train_series.tolist()
    model = _fit_model(np.asarray(history, dtype=float))

    preds = []
    for i in range(len(test_series)):
        if do_refit and i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-int(seq_len) :], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, int(seq_len))

        y_hat_scaled = float(np.asarray(model.predict(win_scaled, verbose=0)).reshape(-1)[0])
        y_hat = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        preds.append(y_hat)
        history.append(float(test_series[i]))

    preds = np.asarray(preds, dtype=float)

    if len(preds) != len(test_series):
        raise RuntimeError(f"NBEATS strict len check failed: preds={len(preds)} vs test={len(test_series)}")

    return preds, test_series


nbeats_predict = nbeats
