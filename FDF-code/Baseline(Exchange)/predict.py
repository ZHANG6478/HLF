import sys
import os
import numpy as np
import inspect
import random
import pandas as pd

SEED = 38
random.seed(SEED)
np.random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from common import (
    setup_chinese_display,
    load_exchange_series,
    calculate_metrics,
    plot_predictions,
    save_pred_table,
    save_metrics_tables,
)

from model.ARIMA import arima_predict
from model.Prophet import prophet_predict
from model.RandomForest import rf_predict
from model.XGBoost import xgb_predict
from model.LightGBM import lgb_predict
from model.LSTM import lstm_predict
from model.GRU import gru_predict
from model.CNN import cnn_predict
from model.TCN import tcn_predict
from model.Transformer import transformer_predict
from model.PatchTST import patchtst_predict
from model.DLinear import dlinear_predict
from model.NBEATS import nbeats_predict

SEQ_LEN = 96
EPOCHS = 20
REFIT_INTERVAL = None

MODEL_CONFIG = [
    {"name": "ARIMA", "func": arima_predict},
    {"name": "Prophet", "func": prophet_predict},
    {"name": "RandomForest", "func": rf_predict},
    {"name": "XGBoost", "func": xgb_predict},
    {"name": "LightGBM", "func": lgb_predict},
    {"name": "LSTM", "func": lstm_predict},
    {"name": "GRU", "func": gru_predict},
    {"name": "CNN", "func": cnn_predict},
    {"name": "TCN", "func": tcn_predict},
    {"name": "Transformer", "func": transformer_predict},
    {"name": "PatchTST", "func": patchtst_predict},
    {"name": "DLinear", "func": dlinear_predict},
    {"name": "N-BEATS", "func": nbeats_predict},
]

NAN_METRICS = {
    "MAE": np.nan,
    "MSE": np.nan,
    "RMSE": np.nan,
    "MAPE(%)": np.nan,
    "SMAPE(%)": np.nan,
    "DA(%)": np.nan,
    "IA": np.nan,
    "R": np.nan,
    "R²": np.nan,
}


def _call_model(model_func, train_data, test_data, train_dates, test_dates):
    sig = inspect.signature(model_func)
    kwargs = {}

    if "seq_len" in sig.parameters:
        kwargs["seq_len"] = SEQ_LEN
    if "epochs" in sig.parameters:
        kwargs["epochs"] = EPOCHS
    if "train_dates" in sig.parameters:
        kwargs["train_dates"] = train_dates
    if "test_dates" in sig.parameters:
        kwargs["test_dates"] = test_dates
    if "refit_interval" in sig.parameters:
        kwargs["refit_interval"] = REFIT_INTERVAL

    out = model_func(train_data=train_data, test_data=test_data, **kwargs)
    if isinstance(out, tuple):
        return np.asarray(out[0]), np.asarray(out[1])
    return np.asarray(out), np.asarray(test_data)


def main():
    setup_chinese_display()

    per_col_metrics = {}
    exchange_cols = [str(i) for i in range(3, 4)]

    for col in exchange_cols:
        print("\n" + "=" * 80)
        print(f"Exchange-{col} forecasting")
        print("=" * 80)

        data = load_exchange_series("exchange_rate.csv", col)
        col_metrics = {}

        for cfg in MODEL_CONFIG:
            name = cfg["name"]
            func = cfg["func"]
            print(f"\nRunning {name} on Exchange-{col}")

            try:
                preds, y_true = _call_model(
                    func,
                    data["train_data"],
                    data["test_data"],
                    data["train_dates"],
                    data["test_dates"],
                )

                n = min(len(preds), len(y_true))
                preds = preds[:n]
                y_true = y_true[:n]
                dates = data["test_dates"][:n]

                metrics = calculate_metrics(y_true, preds, data["train_min"], data["train_max"])
                col_metrics[name] = metrics

                plot_predictions(
                    dates,
                    y_true,
                    preds,
                    model_name=f"{name}-Ex{col}",
                    title_suffix="Exchange",
                )

                save_pred_table(
                    dates,
                    y_true,
                    preds,
                    model_name=f"{name}-Ex{col}",
                    title_suffix="Exchange",
                )

                print(metrics)

            except Exception as e:
                print(f"{name} failed: {e}")
                col_metrics[name] = dict(NAN_METRICS)

        per_col_metrics[col] = pd.DataFrame(col_metrics).T

    all_models = [cfg["name"] for cfg in MODEL_CONFIG]
    avg_metrics_dict = {}

    for model in all_models:
        model_rows = []
        for col in exchange_cols:
            if model in per_col_metrics[col].index:
                model_rows.append(per_col_metrics[col].loc[model].to_dict())

        if model_rows:
            avg_metrics_dict[model] = pd.DataFrame(model_rows).mean().round(6).to_dict()
        else:
            avg_metrics_dict[model] = dict(NAN_METRICS)

    average_metrics = pd.DataFrame(avg_metrics_dict).T
    save_metrics_tables(per_col_metrics, average_metrics)

    print("\nAll tasks finished. Outputs saved to result/.")


if __name__ == "__main__":
    main()
