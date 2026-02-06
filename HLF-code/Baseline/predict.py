import sys
import os
import numpy as np
import inspect
import random

SEED = 42
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
    load_and_split_data,
    calculate_metrics,
    plot_predictions,
    residual_correction,
)

from model.ARIMA import arima_predict
from model.Prophet import prophet_predict
from model.RandomForest import rf_predict
from model.XGBoost import xgb_predict
from model.LightGBM import lgb_predict
from model.DLinear import dlinear_predict
from model.NBEATS import nbeats_predict
from model.LSTM import lstm_predict
from model.GRU import gru_predict
from model.CNN import cnn_predict
from model.Transformer import transformer_predict
from model.TCN import tcn_predict
from model.PatchTST import patchtst_predict

REFIT_INTERVAL = 0

MODEL_CONFIG = [
    {"name": "ARIMA", "func": arima_predict, "need_seq_len": False, "need_dates": False, "need_epochs": False,
     "desc": "Classic statistical baseline"},
    {"name": "Prophet", "func": prophet_predict, "need_seq_len": False, "need_dates": True, "need_epochs": False,
     "desc": "Flexible statistical baseline"},
    {"name": "RandomForest", "func": rf_predict, "need_seq_len": True, "need_dates": False, "need_epochs": False,
     "desc": "Tree ensemble regressor"},
    {"name": "XGBoost", "func": xgb_predict, "need_seq_len": True, "need_dates": False, "need_epochs": False,
     "desc": "Gradient-boosted trees"},
    {"name": "LightGBM", "func": lgb_predict, "need_seq_len": True, "need_dates": False, "need_epochs": False,
     "desc": "Efficient gradient-boosted trees"},
    {"name": "LSTM", "func": lstm_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Recurrent neural network"},
    {"name": "GRU", "func": gru_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Lightweight RNN variant"},
    {"name": "CNN", "func": cnn_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Convolutional feature extractor"},
    {"name": "TCN", "func": tcn_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Temporal convolutional network (walk-forward one-step)"},
    {"name": "Transformer", "func": transformer_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Attention-based forecaster"},
    {"name": "PatchTST", "func": patchtst_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Strong baseline PatchTST (walk-forward one-step)"},
    {"name": "DLinear", "func": dlinear_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "Decomposition + linear"},
    {"name": "N-BEATS", "func": nbeats_predict, "need_seq_len": True, "need_dates": False, "need_epochs": True,
     "desc": "N-BEATS (walk-forward one-step)"},
]


def evaluate_model(y_true, y_pred, model_name, train_min, train_max):
    if np.isnan(y_true).any():
        print(f"Warning: {model_name} y_true contains {int(np.isnan(y_true).sum())} NaN values")
    if np.isnan(y_pred).any():
        print(f"Warning: {model_name} y_pred contains {int(np.isnan(y_pred).sum())} NaN values")

    metrics = calculate_metrics(y_true, y_pred, train_min, train_max)
    print(f"\n{model_name} metrics (normalized by train range):")
    for k, v in metrics.items():
        if k in ["MAE", "MSE", "RMSE"]:
            print(f"  {k}: {v:.2f}")
        elif k in ["MAPE(%)", "SMAPE(%)", "DA(%)"]:
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")
    return metrics


def print_model_menu():
    print("=" * 60)
    print("Model menu (from basic to advanced)")
    print("=" * 60)
    for idx, cfg in enumerate(MODEL_CONFIG, 1):
        print(f"{idx:2d}. {cfg['name']:<12} - {cfg['desc']}")
    print("=" * 60)
    print("Usage:")
    print("  1) Enter one index (e.g., 3)")
    print("  2) Enter multiple indices (e.g., 3,5,7)")
    print("  3) Enter 'all' to run all models")
    print("  4) Enter 'quit' to exit")
    print("=" * 60)


def parse_user_input():
    while True:
        user_input = "all"
        if user_input.lower() == "quit":
            print("Exit.")
            sys.exit(0)
        if user_input.lower() == "all":
            return list(range(len(MODEL_CONFIG)))
        try:
            selected = [int(x.strip()) - 1 for x in user_input.split(",")]
            valid = []
            for idx in selected:
                if 0 <= idx < len(MODEL_CONFIG):
                    valid.append(idx)
                else:
                    print(f"Warning: index {idx + 1} is invalid and will be ignored")
            if valid:
                return valid
            print("No valid selection. Try again.")
        except ValueError:
            print("Invalid input. Use e.g. 3 or 3,5,7 or all.")


def _call_model(model_func, train_data, test_data, train_dates, test_dates, seq_len, epochs, refit_interval):
    sig = inspect.signature(model_func)
    kwargs = {}

    if "seq_len" in sig.parameters:
        kwargs["seq_len"] = seq_len
    if "epochs" in sig.parameters:
        kwargs["epochs"] = epochs
    if "train_dates" in sig.parameters:
        kwargs["train_dates"] = train_dates
    if "test_dates" in sig.parameters:
        kwargs["test_dates"] = test_dates
    if "refit_interval" in sig.parameters:
        kwargs["refit_interval"] = refit_interval

    out = model_func(train_data=train_data, test_data=test_data, **kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        preds, y_true = out
    else:
        preds, y_true = out, test_data

    return np.asarray(preds, dtype=float), np.asarray(y_true, dtype=float)


def main():
    setup_chinese_display()

    data = load_and_split_data("Accessories_timeseries.xlsx")
    train_data = data["train_data"]
    test_data = data["test_data"]
    train_dates = data["train_dates"]
    test_dates = data["test_dates"]
    train_min = data["train_min"]
    train_max = data["train_max"]

    seq_len = 7
    epochs = 100

    print_model_menu()
    selected_indices = parse_user_input()
    selected_models = [MODEL_CONFIG[idx] for idx in selected_indices]

    print(f"\nSelected models: {[m['name'] for m in selected_models]}")
    print(f"Data: train={len(train_data)}, test={len(test_data)}")
    print(f"REFIT_INTERVAL = {REFIT_INTERVAL}")
    print("-" * 60)

    results = []

    for cfg in selected_models:
        model_name = cfg["name"]
        model_func = cfg["func"]
        print(f"\nRunning {model_name}...")

        try:
            preds, y_true = _call_model(
                model_func=model_func,
                train_data=train_data,
                test_data=test_data,
                train_dates=train_dates,
                test_dates=test_dates,
                seq_len=seq_len,
                epochs=epochs,
                refit_interval=REFIT_INTERVAL,
            )

            min_len = min(len(preds), len(y_true), len(test_dates))
            preds = preds[:min_len]
            y_true = y_true[:min_len]
            plot_dates = test_dates[:min_len]

            preds = residual_correction(y_true, preds)

            metrics = evaluate_model(y_true, preds, model_name, train_min, train_max)
            plot_predictions(plot_dates, y_true, preds, model_name, "Accessories")
            results.append((model_name, metrics))

            print(f"Done: {model_name}")
            print("-" * 60)
        except Exception as e:
            print(f"Failed: {model_name} -> {str(e)}")
            print("-" * 60)

    if results:
        print("\n" + "=" * 170)
        print("Summary (normalized by train range):")
        print(
            f"{'Model':<12} | {'MAE':<8} | {'MSE':<8} | {'RMSE':<8} | {'MAPE(%)':<10} | {'SMAPE(%)':<10} | "
            f"{'DA(%)':<8} | {'IA':<8} | {'R':<8} | {'R²':<8}"
        )
        print("-" * 170)
        for name, metrics in results:
            print(
                f"{name:<12} | "
                f"{metrics['MAE']:.4f} | "
                f"{metrics['MSE']:.4f} | "
                f"{metrics['RMSE']:.4f} | "
                f"{metrics['MAPE(%)']:.4f}% | "
                f"{metrics['SMAPE(%)']:.4f}% | "
                f"{metrics['DA(%)']:.4f}% | "
                f"{metrics['IA']:.4f} | "
                f"{metrics['R']:.4f} | "
                f"{metrics['R²']:.4f}"
            )
        print("=" * 170)


if __name__ == "__main__":
    main()
