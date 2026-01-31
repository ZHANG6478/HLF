import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Sequence
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _get_best_arima_order(train_data, p_max: int = 4, q_max: int = 4) -> Tuple[int, int, int]:
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

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                fit = ARIMA(train, order=(p, d, q)).fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = float(fit.aic)
                    best_order = (p, d, q)
            except Exception:
                continue

    print(f"Best ARIMA order: {best_order}")
    return best_order


def arima(
    series_train: Sequence[float],
    series_test: Sequence[float],
    order: Optional[Tuple[int, int, int]] = (1, 0, 1),
    refit_interval: Optional[int] = 7,
    p_max: int = 4,
    q_max: int = 4,
    use_drift: bool = True,
    noise_scale: float = 0.0,
    non_negative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.asarray(series_train, dtype=float)
    test_data = np.asarray(series_test, dtype=float)

    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    do_refit = (refit_interval is not None) and (refit_interval > 0)
    if not do_refit:
        refit_interval = len(test_data)

    history = pd.Series(train_data).dropna().astype(float).tolist()
    if len(history) == 0:
        raise ValueError("series_train has no valid (non-NaN) values")

    preds = []
    model_fit = None
    cur_order = None
    steps_since_refit = 0

    def _fit():
        nonlocal model_fit, cur_order

        cur_order = _get_best_arima_order(history, p_max=p_max, q_max=q_max) if order is None else tuple(order)
        d = int(cur_order[1])

        trend = "t" if (use_drift and d > 0) else ("c" if d == 0 else "n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_fit = ARIMA(pd.Series(history), order=cur_order, trend=trend).fit()

    _fit()

    for i in range(len(test_data)):
        if do_refit and steps_since_refit >= int(refit_interval):
            _fit()
            steps_since_refit = 0

        try:
            fc = model_fit.forecast(steps=1)
            y_hat = float(fc.iloc[0]) if hasattr(fc, "iloc") else float(np.asarray(fc).ravel()[0])
            if not np.isfinite(y_hat):
                y_hat = float(history[-1])
        except Exception:
            y_hat = float(history[-1])

        if noise_scale and noise_scale > 0:
            m = float(np.mean(np.asarray(history, dtype=float)))
            y_hat += float(np.random.normal(loc=0.0, scale=noise_scale * (abs(m) + 1e-12), size=1)[0])

        if non_negative:
            y_hat = max(y_hat, 0.0)

        preds.append(float(y_hat))

        y_true = float(test_data[i])
        history.append(y_true)

        try:
            model_fit = model_fit.append([y_true], refit=False)
        except Exception:
            pass

        if do_refit:
            steps_since_refit += 1

    return np.asarray(preds, dtype=float), test_data
