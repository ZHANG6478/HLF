# models/tcn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple

from common import create_sequence_data


class ResidualBlock(layers.Layer):
    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int, dropout: float = 0.1):
        super().__init__()
        self.n_filters = int(n_filters)

        self.conv1 = layers.Conv1D(
            filters=self.n_filters,
            kernel_size=int(kernel_size),
            dilation_rate=int(dilation_rate),
            padding="causal",
        )
        self.act1 = layers.Activation("relu")
        self.drop1 = layers.Dropout(float(dropout))

        self.conv2 = layers.Conv1D(
            filters=self.n_filters,
            kernel_size=int(kernel_size),
            dilation_rate=int(dilation_rate),
            padding="causal",
        )
        self.act2 = layers.Activation("relu")
        self.drop2 = layers.Dropout(float(dropout))

        self.downsample = None
        self.add = layers.Add()
        self.final_act = layers.Activation("relu")

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if in_channels != self.n_filters:
            self.downsample = layers.Conv1D(filters=self.n_filters, kernel_size=1, padding="same")
        super().build(input_shape)

    def call(self, x, training=None):
        y = self.conv1(x)
        y = self.act1(y)
        y = self.drop1(y, training=training)

        y = self.conv2(y)
        y = self.act2(y)
        y = self.drop2(y, training=training)

        res = x if self.downsample is None else self.downsample(x)
        out = self.add([res, y])
        return self.final_act(out)


def _build_tcn(
    seq_len: int,
    n_filters: int = 32,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(int(seq_len), 1))
    x = inputs

    for i in range(int(n_blocks)):
        x = ResidualBlock(
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=2**i,
            dropout=dropout,
        )(x)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(float(dropout))(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TCN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)), loss="mse")
    return model


def tcn(
    train_series,
    test_series,
    seq_len: int = 7,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_filters: int = 32,
    kernel_size: int = 3,
    n_blocks: int = 4,
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

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    refit_interval = int(refit_interval) if do_refit else len(test_series)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.reshape(-1, 1))

    def _fit_model(history_raw: np.ndarray) -> tf.keras.Model:
        hist_scaled = scaler.transform(history_raw.reshape(-1, 1)).flatten()
        X_train, y_train = create_sequence_data(hist_scaled, seq_len)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = _build_tcn(
            seq_len=seq_len,
            n_filters=n_filters,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            dropout=dropout,
            learning_rate=learning_rate,
        )

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

    preds = np.zeros(len(test_series), dtype=float)

    for i in range(len(test_series)):
        if i > 0 and (i % refit_interval == 0):
            model = _fit_model(np.asarray(history, dtype=float))

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        y_hat_scaled = float(model.predict(win_scaled, verbose=0).reshape(-1)[0])
        preds[i] = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        history.append(float(test_series[i]))

    return preds, test_series


tcn_predict = tcn
