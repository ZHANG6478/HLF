import numpy as np
import lightgbm as lgb
from common import create_sequence_data


def lgb_predict(train_data, test_data, seq_len=5, refit_interval=7):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= seq_len:
        raise ValueError(f"train_data length ({len(train_data)}) must be > seq_len ({seq_len})")
    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    def _build_model():
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )

    history = train_data.tolist()
    preds = []

    i = 0
    while i < len(test_data):
        X_train, y_train = create_sequence_data(np.asarray(history, dtype=float), seq_len)
        model = _build_model()
        model.fit(X_train, y_train)

        block_end = min(i + refit_interval, len(test_data))
        for k in range(i, block_end):
            x_win = np.asarray(history[-seq_len:], dtype=float).reshape(1, -1)
            y_hat = float(model.predict(x_win)[0])
            preds.append(y_hat)
            history.append(float(test_data[k]))

        i = block_end

    return np.asarray(preds, dtype=float), test_data
