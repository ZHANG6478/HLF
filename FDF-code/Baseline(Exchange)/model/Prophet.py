import numpy as np
import pandas as pd
from prophet import Prophet


def prophet_predict(
    train_data,
    test_data,
    train_dates,
    test_dates,
    refit_interval=7,
    iter=1000,
):
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    train_dates = pd.to_datetime(pd.Series(train_dates))
    test_dates = pd.to_datetime(pd.Series(test_dates))

    if len(test_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    df_history = pd.DataFrame({"ds": train_dates, "y": train_data}).dropna().reset_index(drop=True)

    def _fit_prophet(df):
        model = Prophet(
            growth="linear",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            n_changepoints=20,
            interval_width=0.95,
        )
        model.fit(df, iter=iter)
        return model

    if refit_interval is None or refit_interval == 0:
        refit_interval = len(test_data)

    preds = []
    i = 0

    while i < len(test_data):
        model = _fit_prophet(df_history)

        block_end = min(i + refit_interval, len(test_data))
        future_dates = pd.DataFrame({"ds": test_dates.iloc[i:block_end].values})
        forecast = model.predict(future_dates)

        block_preds = forecast["yhat"].to_numpy(dtype=float)
        block_preds = np.maximum(block_preds, 0.0)

        for k in range(i, block_end):
            preds.append(float(block_preds[k - i]))
            df_history = pd.concat(
                [df_history, pd.DataFrame({"ds": [test_dates.iloc[k]], "y": [float(test_data[k])]} )],
                ignore_index=True,
            )

        i = block_end

    return np.asarray(preds, dtype=float), test_data
