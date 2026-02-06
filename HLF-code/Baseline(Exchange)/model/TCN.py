import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from common import create_sequence_data


class ResidualBlock(layers.Layer):
    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int, dropout: float = 0.1):
        super().__init__()
        self.n_filters = n_filters

        self.conv1 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
        )
        self.act1 = layers.Activation("relu")
        self.drop1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
        )
        self.act2 = layers.Activation("relu")
        self.drop2 = layers.Dropout(dropout)

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


def _build_tcn_paper(
    seq_len: int,
    n_filters: int = 32,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
):
    inputs = layers.Input(shape=(seq_len, 1))
    x = inputs

    for i in range(n_blocks):
        x = ResidualBlock(
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,
            dropout=dropout,
        )(x)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TCN_Paper")
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def tcn_predict(
    train_data,
    test_data,
    seq_len=7,
    epochs=80,
    batch_size=32,
    learning_rate=1e-3,
    n_filters=32,
    kernel_size=3,
    n_blocks=4,
    dropout=0.1,
    refit_interval=7,
):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= seq_len:
        raise ValueError(f"train_data length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.reshape(-1, 1))

    def _fit_model(current_history_raw: np.ndarray):
        hist_scaled = scaler.transform(current_history_raw.reshape(-1, 1)).flatten()

        X_train, y_train = create_sequence_data(hist_scaled, seq_len)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = _build_tcn_paper(
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

        y_hat_scaled = model.predict(win_scaled, verbose=0).flatten()[0]
        y_hat = scaler.inverse_transform([[y_hat_scaled]])[0, 0]

        preds.append(float(y_hat))
        history.append(float(test_data[i]))

    return np.asarray(preds, dtype=float), test_data
