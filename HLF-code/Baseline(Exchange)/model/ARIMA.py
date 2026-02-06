import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _get_best_arima_order(train_data):
    train = pd.Series(train_data).dropna().astype(float)

    d = 0
    try:
        if adfuller(train)[1] >= 0.05:
            d = 1
            if adfuller(train.diff().dropna())[1] >= 0.05:
                d = 2
    except Exception:
        d = 1

    best_aic = float("inf")
    best_order = (0, d, 0)

    for p in range(0, 5):
        for q in range(0, 5):
            try:
                fit = ARIMA(train, order=(p, d, q)).fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    print(f"Best ARIMA order: {best_order}")
    return best_order


def arima_predict(train_data, test_data, order=None, refit_interval=7):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    history = pd.Series(train_data).dropna().astype(float).tolist()
    preds = []

    i = 0
    while i < len(test_data):
        cur_order = _get_best_arima_order(history) if order is None else order

        model_fit = ARIMA(pd.Series(history), order=cur_order).fit()

        block_end = min(i + refit_interval, len(test_data))
        steps = block_end - i

        block_pred = model_fit.forecast(steps=steps)
        if hasattr(block_pred, "values"):
            block_pred = block_pred.values
        block_pred = np.asarray(block_pred, dtype=float)

        preds.extend(block_pred.tolist())
        history.extend(test_data[i:block_end].astype(float).tolist())

        i = block_end

    preds = np.asarray(preds, dtype=float)
    return preds, test_data
