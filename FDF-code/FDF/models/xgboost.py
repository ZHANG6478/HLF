# models/xgboost.py
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple, Optional, Dict, Any


def create_ts_features(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    for lag in range(1, 15):
        df[f"lag_{lag}"] = df[value_col].shift(lag)

    df["roll_mean_7"] = df[value_col].rolling(7).mean()
    df["roll_std_7"] = df[value_col].rolling(7).std()
    df["roll_mean_14"] = df[value_col].rolling(14).mean()

    return df.dropna()


def _make_next_feat(next_date: pd.Timestamp, hist_values: np.ndarray) -> pd.DataFrame:
    last_series = np.asarray(hist_values, dtype=float)

    feat: Dict[str, Any] = {
        "year": int(next_date.year),
        "month": int(next_date.month),
        "day": int(next_date.day),
        "weekday": int(next_date.weekday()),
        "is_weekend": 1 if next_date.weekday() in [5, 6] else 0,
    }

    for lag in range(1, 15):
        feat[f"lag_{lag}"] = float(last_series[-lag]) if lag <= len(last_series) else float(last_series.mean())

    feat["roll_mean_7"] = float(last_series[-7:].mean()) if len(last_series) >= 7 else float(last_series.mean())
    feat["roll_std_7"] = float(last_series[-7:].std()) if len(last_series) >= 7 else 0.0
    feat["roll_mean_14"] = float(last_series[-14:].mean()) if len(last_series) >= 14 else float(last_series.mean())

    return pd.DataFrame([feat])


def xgboost(
    train_dates,
    train_series,
    test_dates,
    test_series,
    refit_interval: Optional[int] = 7,
    min_history: int = 30,
    xgb_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    train_series = np.asarray(train_series, dtype=float)
    test_series = np.asarray(test_series, dtype=float)

    train_dates = pd.to_datetime(pd.Series(train_dates))
    test_dates = pd.to_datetime(pd.Series(test_dates))

    n_test = len(test_series)
    if n_test == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if xgb_params is None:
        xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "verbosity": 0,
            "random_state": 42,
        }

    hist_dates = list(train_dates)
    hist_values = list(train_series)

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    refit_steps = int(refit_interval) if do_refit else n_test

    model: Optional[xgb.XGBRegressor] = None

    def _fit_current_model() -> None:
        nonlocal model
        if len(hist_values) < int(min_history):
            model = None
            return

        df_history = pd.DataFrame({"ds": pd.to_datetime(hist_dates), "y": np.asarray(hist_values, dtype=float)})
        df_feat = create_ts_features(df_history)
        if len(df_feat) < 10:
            model = None
            return

        X = df_feat.drop(["ds", "y"], axis=1)
        y = df_feat["y"]

        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X, y)
        model = m

    _fit_current_model()
    steps_since_refit = 0

    preds = np.zeros(n_test, dtype=float)

    for i in range(n_test):
        if do_refit and (model is None or steps_since_refit >= refit_steps):
            _fit_current_model()
            steps_since_refit = 0

        if model is None:
            y_hat = float(hist_values[-1])
        else:
            next_date = pd.to_datetime(test_dates.iloc[i])
            X_next = _make_next_feat(next_date, np.asarray(hist_values, dtype=float))
            y_hat = float(model.predict(X_next)[0])

        preds[i] = y_hat

        hist_dates.append(test_dates.iloc[i])
        hist_values.append(float(test_series[i]))

        if do_refit:
            steps_since_refit += 1

    return preds, test_series


xgboost_predict = xgboost
