import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from common import create_sequence_data


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def positional_encoding(length, embed_dim):
    position = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    pos_encoding = np.zeros((length, embed_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


def _build_transformer(seq_len, embed_dim, num_heads, ff_dim, learning_rate):
    inputs = layers.Input(shape=(seq_len, 1))
    x = layers.Dense(embed_dim)(inputs)

    x = x + positional_encoding(seq_len, embed_dim)

    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def transformer_predict(
    train_data,
    test_data,
    seq_len=5,
    embed_dim=32,
    num_heads=2,
    ff_dim=32,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
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

        model = _build_transformer(
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            learning_rate=learning_rate,
        )

        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es],
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
