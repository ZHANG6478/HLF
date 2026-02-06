# -*- coding: utf-8 -*-

import os
import inspect
from typing import Tuple, Dict, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import calculate_metrics, plot_predictions, plot_fusion_result
from utils import static_weight_fusion, simple_residual_correction
from models import LOW_MODEL_REGISTRY, HIGH_MODEL_REGISTRY


def _to_float_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def _strict_len_check(preds: np.ndarray, y_true: np.ndarray, expected_len: int, model_name: str) -> None:
    if len(preds) != expected_len:
        raise ValueError(
            f"[{model_name}] predictor must output len(test)={expected_len}, got len(preds)={len(preds)}. "
            "Use walk-forward one-step starting from test[0]."
        )
    if len(y_true) != expected_len:
        raise ValueError(
            f"[{model_name}] predictor must output y_true len(test)={expected_len}, got len(y_true)={len(y_true)}."
        )


def _call_predictor(
    predict_func: Callable,
    *,
    train_dates,
    train_series,
    test_dates,
    test_series,
    seq_len: int,
    epochs: Optional[int] = None,
    refit_interval: Optional[int] = 7,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    sig = inspect.signature(predict_func)
    kwargs = {}

    if "train_dates" in sig.parameters:
        kwargs["train_dates"] = train_dates
    if "test_dates" in sig.parameters:
        kwargs["test_dates"] = test_dates

    if "train_series" in sig.parameters:
        kwargs["train_series"] = train_series
    if "test_series" in sig.parameters:
        kwargs["test_series"] = test_series

    if "series_train" in sig.parameters:
        kwargs["series_train"] = train_series
    if "series_test" in sig.parameters:
        kwargs["series_test"] = test_series

    if "seq_len" in sig.parameters:
        kwargs["seq_len"] = int(seq_len)
    if epochs is not None and "epochs" in sig.parameters:
        kwargs["epochs"] = epochs
    if "verbose" in sig.parameters:
        kwargs["verbose"] = bool(verbose)
    if "refit_interval" in sig.parameters:
        kwargs["refit_interval"] = refit_interval

    out = predict_func(**kwargs)

    if not isinstance(out, tuple) or len(out) < 2:
        raise ValueError(f"{predict_func.__name__} must return (preds, y_true).")

    preds = _to_float_1d(out[0])
    y_true = _to_float_1d(out[1])

    expected_len = len(test_series)
    _strict_len_check(preds, y_true, expected_len, predict_func.__name__)
    return preds, y_true


def vmd_xgboost_gru_fusion_forecast(
    modes,
    omega,
    all_dates,
    all_data,
    train_size: int,
    low_modes_idx,
    high_modes_idx,
    seq_len: int = 96,
    energy_threshold: float = 0.9,
    save_prefix: str = "Exchange",
    train_min: float = None,
    train_max: float = None,
    refit_interval: Optional[int] = None,
    low_model_name: str = "XGBoost",
    high_model_name: str = "GRU",
    low_epochs: Optional[int] = None,
    high_epochs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    all_data = _to_float_1d(all_data)
    all_dates = np.asarray(all_dates)
    modes = np.asarray(modes, dtype=float)

    test_data = all_data[int(train_size) :]
    if len(test_data) == 0:
        return pd.DataFrame(), {}

    train_dates = all_dates[: int(train_size)]
    test_dates = all_dates[int(train_size) :]
    modes_train = modes[:, : int(train_size)]
    modes_test = modes[:, int(train_size) :]

    low_idx0 = [int(m) - 1 for m in low_modes_idx]
    high_idx0 = [int(m) - 1 for m in high_modes_idx]

    if low_model_name not in LOW_MODEL_REGISTRY:
        raise ValueError(f"Unknown low_model_name={low_model_name}.")
    if high_model_name not in HIGH_MODEL_REGISTRY:
        raise ValueError(f"Unknown high_model_name={high_model_name}.")

    low_predict_func = LOW_MODEL_REGISTRY[low_model_name]
    high_predict_func = HIGH_MODEL_REGISTRY[high_model_name]

    low_preds: Dict[int, np.ndarray] = {}
    for idx in low_idx0:
        preds, _ = _call_predictor(
            low_predict_func,
            train_dates=train_dates,
            train_series=modes_train[idx],
            test_dates=test_dates,
            test_series=modes_test[idx],
            seq_len=seq_len,
            epochs=low_epochs,
            refit_interval=refit_interval,
            verbose=verbose,
        )
        low_preds[idx] = preds

    low_pred = np.sum(list(low_preds.values()), axis=0) if low_preds else np.zeros(len(test_data), dtype=float)

    high_preds: Dict[int, np.ndarray] = {}
    for idx in high_idx0:
        preds, _ = _call_predictor(
            high_predict_func,
            train_dates=train_dates,
            train_series=modes_train[idx],
            test_dates=test_dates,
            test_series=modes_test[idx],
            seq_len=seq_len,
            epochs=high_epochs,
            refit_interval=refit_interval,
            verbose=verbose,
        )
        high_preds[idx] = preds

    high_pred_sum = np.sum(list(high_preds.values()), axis=0) if high_preds else np.zeros(len(test_data), dtype=float)

    if low_idx0 or high_idx0:
        target_modes = modes[low_idx0 + high_idx0]
    else:
        target_modes = modes
    modal_true_test = np.sum(target_modes, axis=0)[int(train_size) : int(train_size) + len(test_data)]

    final_pred = static_weight_fusion(low_pred, high_preds)
    final_pred = _to_float_1d(final_pred)
    _strict_len_check(final_pred, modal_true_test, len(test_data), "Fusion")

    final_pred = simple_residual_correction(modal_true_test, final_pred)

    true_sales = all_data[int(train_size) : int(train_size) + len(test_data)]
    true_sales_dates = all_dates[int(train_size) : int(train_size) + len(test_data)]
    _strict_len_check(final_pred, true_sales, len(test_data), "FinalPrediction")

    metrics = calculate_metrics(true_sales, final_pred, train_min, train_max) if len(true_sales) else {}

    for idx in low_idx0:
        mode_all = modes[idx]
        mode_true_all = np.concatenate(
            [mode_all[: int(train_size)], mode_all[int(train_size) : int(train_size) + len(test_data)]]
        )
        mode_pred_padded = np.concatenate([np.full(int(train_size), np.nan), low_preds[idx]])
        mode_dates = np.concatenate([train_dates, test_dates])

        plot_predictions(
            dates=mode_dates,
            y_true=mode_true_all,
            y_pred=mode_pred_padded,
            model_name=f"Mode_{idx + 1}",
            title_suffix=low_model_name,
            train_min=train_min,
            train_max=train_max,
            category=save_prefix,
        )

    for idx in high_idx0:
        mode_all = modes[idx]
        mode_true_all = np.concatenate(
            [mode_all[: int(train_size)], mode_all[int(train_size) : int(train_size) + len(test_data)]]
        )
        mode_pred_padded = np.concatenate([np.full(int(train_size), np.nan), high_preds[idx]])
        mode_dates = np.concatenate([train_dates, test_dates])

        plot_predictions(
            dates=mode_dates,
            y_true=mode_true_all,
            y_pred=mode_pred_padded,
            model_name=f"Mode_{idx + 1}",
            title_suffix=high_model_name,
            train_min=train_min,
            train_max=train_max,
            category=save_prefix,
        )

    true_all = np.concatenate([all_data[: int(train_size)], true_sales])
    final_pred_padded = np.concatenate([np.full(int(train_size), np.nan), final_pred])
    all_dates_plot = np.concatenate([all_dates[: int(train_size)], true_sales_dates])

    plot_fusion_result(
        dates=all_dates_plot,
        y_true=true_all,
        y_pred=final_pred_padded,
        save_prefix=save_prefix,
        train_min=train_min,
        train_max=train_max,
    )

    result_df = pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in true_sales_dates] if len(true_sales_dates) else [],
            "Actual_Sales": true_sales,
            "Final_Prediction": final_pred,
            "LowFreq_Sum_Prediction": low_pred,
            "HighFreq_Sum_Prediction": high_pred_sum,
            "Low_Model": low_model_name,
            "High_Model": high_model_name,
            "refit_interval": refit_interval if refit_interval is not None else 0,
        }
    )

    for idx in low_preds:
        result_df[f"Low_Mode_{idx + 1}_Pred"] = low_preds[idx]
    for idx in high_preds:
        result_df[f"High_Mode_{idx + 1}_Pred"] = high_preds[idx]

    save_dir = os.path.join("result")
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"{save_prefix}_Prediction_Result.csv")
    result_df.to_csv(out_csv, index=False, float_format="%.6f", encoding="utf-8")

    if len(true_sales) and len(final_pred):
        plt.rcParams.update(
            {
                "font.family": "Times New Roman",
                "font.size": 12,
                "axes.linewidth": 0.6,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.unicode_minus": False,
                "grid.alpha": 0.3,
            }
        )

        plt.figure(figsize=(6, 4))
        plt.plot(true_sales_dates, true_sales, label="True Value", color=(0.5, 0.5, 1.0), linewidth=1.5)
        plt.plot(
            true_sales_dates,
            final_pred,
            label="Predicted Value",
            color=(1.0, 0.5, 0.5),
            linestyle="--",
            linewidth=1.5,
        )
        plt.title(f"{save_prefix} Sales Prediction ({low_model_name}+{high_model_name})", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10, frameon=True)
        plt.grid()
        plt.tight_layout()

        predict_dir = os.path.join("result")
        os.makedirs(predict_dir, exist_ok=True)
        save_path = os.path.join(predict_dir, f"{save_prefix}_Prediction_Result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.figure(figsize=(6, 3))
        plt.plot(
            true_sales_dates,
            true_sales - final_pred,
            marker="o",
            color=(0.2, 0.5, 0.7),
            linewidth=1.0,
            markersize=4,
        )
        plt.axhline(y=0, color=(0.4, 0.4, 0.4), linestyle="--", linewidth=1.0)
        plt.title(f"{save_prefix} Fusion Residuals", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.tight_layout()

        residual_dir = os.path.join("result", "Modal_Analysis", "Fusion_Prediction_Result")
        os.makedirs(residual_dir, exist_ok=True)
        save_path = os.path.join(residual_dir, f"{save_prefix}_Fusion_Residuals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.rcParams.update(plt.rcParamsDefault)

    return result_df, metrics


def evaluate_low_modes_metrics(
    modes: np.ndarray,
    all_dates: np.ndarray,
    train_size: int,
    low_modes_idx: list,
    low_model_name: str = "XGBoost",
    seq_len: int = 14,
    low_epochs: int = None,
    refit_interval: int = 7,
    train_min: float = None,
    train_max: float = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if train_min is None or train_max is None:
        raise ValueError("train_min/train_max must be provided.")

    modes = np.asarray(modes, dtype=float)
    all_dates = np.asarray(all_dates)

    modes_train = modes[:, : int(train_size)]
    modes_test = modes[:, int(train_size) :]
    train_dates = all_dates[: int(train_size)]
    test_dates = all_dates[int(train_size) :]

    if len(test_dates) == 0:
        return pd.DataFrame()

    low_idx0 = [int(m) - 1 for m in low_modes_idx]

    if low_model_name not in LOW_MODEL_REGISTRY:
        raise ValueError(f"Unknown low_model_name={low_model_name}.")

    low_predict_func = LOW_MODEL_REGISTRY[low_model_name]

    rows = []
    for idx in low_idx0:
        preds, y_true = _call_predictor(
            low_predict_func,
            train_dates=train_dates,
            train_series=modes_train[idx],
            test_dates=test_dates,
            test_series=modes_test[idx],
            seq_len=seq_len,
            epochs=low_epochs,
            refit_interval=refit_interval,
            verbose=verbose,
        )

        m = calculate_metrics(y_true, preds, train_min, train_max)
        rows.append(
            {
                "Mode_ID": idx + 1,
                "Low_Model": low_model_name,
                "refit_interval": 0 if (refit_interval is None) else refit_interval,
                **m,
            }
        )

    return pd.DataFrame(rows)


def evaluate_high_modes_metrics(
    modes: np.ndarray,
    all_dates: np.ndarray,
    train_size: int,
    high_modes_idx: list,
    high_model_name: str = "GRU",
    seq_len: int = 14,
    high_epochs: int = None,
    refit_interval: int = 7,
    train_min: float = None,
    train_max: float = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if train_min is None or train_max is None:
        raise ValueError("train_min/train_max must be provided.")

    modes = np.asarray(modes, dtype=float)
    all_dates = np.asarray(all_dates)

    modes_train = modes[:, : int(train_size)]
    modes_test = modes[:, int(train_size) :]
    train_dates = all_dates[: int(train_size)]
    test_dates = all_dates[: int(train_size) + len(modes_test[0])][int(train_size) :]

    if len(test_dates) == 0:
        return pd.DataFrame()

    high_idx0 = [int(m) - 1 for m in high_modes_idx]

    if high_model_name not in HIGH_MODEL_REGISTRY:
        raise ValueError(f"Unknown high_model_name={high_model_name}.")

    high_predict_func = HIGH_MODEL_REGISTRY[high_model_name]

    rows = []
    for idx in high_idx0:
        preds, y_true = _call_predictor(
            high_predict_func,
            train_dates=train_dates,
            train_series=modes_train[idx],
            test_dates=all_dates[int(train_size) :],
            test_series=modes_test[idx],
            seq_len=seq_len,
            epochs=high_epochs,
            refit_interval=refit_interval,
            verbose=verbose,
        )

        m = calculate_metrics(y_true, preds, train_min, train_max)
        rows.append(
            {
                "Mode_ID": idx + 1,
                "High_Model": high_model_name,
                "refit_interval": 0 if (refit_interval is None) else refit_interval,
                **m,
            }
        )

    return pd.DataFrame(rows)
