import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler


def _create_sequence_data_1d(series_1d: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(series_1d)):
        win = series_1d[i - seq_len:i]
        target = series_1d[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Cannot create valid sequences (seq_len={seq_len})")
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _build_cnn(seq_len: int):
    model = tf.keras.Sequential([
        layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(seq_len, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def cnn_predict(train_data, test_data, seq_len=5, epochs=50, batch_size=32, refit_interval=7):
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

        X_train, y_train = _create_sequence_data_1d(hist_scaled, seq_len)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = _build_cnn(seq_len)

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
