import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler


class NBeatsBlock(layers.Layer):
    def __init__(self, input_size, theta_size=128, hidden_units=256, n_hidden_layers=2, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size

        self.hidden = []
        for _ in range(n_hidden_layers):
            self.hidden.append(layers.Dense(hidden_units, activation="relu"))
            if dropout and dropout > 0:
                self.hidden.append(layers.Dropout(dropout))

        self.theta = layers.Dense(theta_size, activation=None)
        self.backcast = layers.Dense(input_size, activation=None)
        self.forecast = layers.Dense(1, activation=None)

    def call(self, x, training=False):
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


def _build_nbeats(input_size, n_stacks=2, n_blocks=3, theta_size=128,
                  hidden_units=256, n_hidden_layers=2, dropout=0.0):
    inp = layers.Input(shape=(input_size,), name="input_window")
    residual = inp
    forecast_sum = 0.0

    for _ in range(n_stacks):
        for _ in range(n_blocks):
            block = NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                hidden_units=hidden_units,
                n_hidden_layers=n_hidden_layers,
                dropout=dropout,
            )
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

    return tf.keras.Model(inputs=inp, outputs=forecast_sum, name="NBEATS")


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


def nbeats_predict(
    train_data,
    test_data,
    seq_len=7,
    epochs=80,
    batch_size=32,
    learning_rate=1e-3,
    refit_interval=0,
    n_stacks=2,
    n_blocks=3,
    theta_size=128,
    hidden_units=256,
    n_hidden_layers=2,
    dropout=0.1,
):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= seq_len:
        raise ValueError(f"train_data length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    def _fit_model(current_history_raw: np.ndarray):
        hist_scaled = scaler.transform(current_history_raw.reshape(-1, 1)).flatten()
        X_train, y_train = _make_supervised(hist_scaled, seq_len)

        model = _build_nbeats(
            input_size=seq_len,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            theta_size=theta_size,
            hidden_units=hidden_units,
            n_hidden_layers=n_hidden_layers,
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
    preds = []

    model = _fit_model(np.asarray(history, dtype=float))

    for i in range(len(test_data)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, -1)

        y_hat_scaled = float(model.predict(win_scaled, verbose=0).flatten()[0])
        y_hat = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        preds.append(y_hat)
        history.append(float(test_data[i]))

    return np.asarray(preds, dtype=float), test_data
