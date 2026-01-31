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
    return np.asarray(x, dtype=float).reshape(-1)


def _strict_len_check(preds: np.ndarray, y_true: np.ndarray, expected_len: int, model_name: str):
    if len(preds) != expected_len:
        raise ValueError(
            f"[{model_name}] predictor must output len(test)={expected_len}, got len(preds)={len(preds)}"
        )
    if len(y_true) != expected_len:
        raise ValueError(
            f"[{model_name}] predictor must output y_true len(test)={expected_len}, got len(y_true)={len(y_true)}"
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
        kwargs["seq_len"] = seq_len
    if epochs is not None and "epochs" in sig.parameters:
        kwargs["epochs"] = epochs
    if "verbose" in sig.parameters:
        kwargs["verbose"] = verbose
    if "refit_interval" in sig.parameters:
        kwargs["refit_interval"] = refit_interval

    out = predict_func(**kwargs)
    if not isinstance(out, tuple) or len(out) < 2:
        raise ValueError(f"{predict_func.__name__} must return (preds, y_true)")

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
    seq_len: int = 14,
    gru_epochs: int = 100,
    energy_threshold: float = 0.9,
    save_prefix: str = "FDF",
    train_min: float = None,
    train_max: float = None,
    refit_interval: Optional[int] = 7,
    low_model_name: str = "XGBoost",
    high_model_name: str = "GRU",
    low_epochs: Optional[int] = None,
    high_epochs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    all_data = _to_float_1d(all_data)
    all_dates = np.asarray(all_dates)
    modes = np.asarray(modes, dtype=float)

    train_data, test_data = all_data[:train_size], all_data[train_size:]
    train_dates, test_dates = all_dates[:train_size], all_dates[train_size:]
    modes_train, modes_test = modes[:, :train_size], modes[:, train_size:]

    print(f"Mode split: train={modes_train.shape} test={modes_test.shape}")
    if len(test_data) == 0:
        print("No test data")
        return pd.DataFrame(), {}

    low_idx0 = [int(m) - 1 for m in low_modes_idx]
    high_idx0 = [int(m) - 1 for m in high_modes_idx]

    if low_model_name not in LOW_MODEL_REGISTRY:
        raise ValueError(f"Unknown low_model_name={low_model_name}. Available: {list(LOW_MODEL_REGISTRY.keys())}")
    if high_model_name not in HIGH_MODEL_REGISTRY:
        raise ValueError(f"Unknown high_model_name={high_model_name}. Available: {list(HIGH_MODEL_REGISTRY.keys())}")

    low_predict_func = LOW_MODEL_REGISTRY[low_model_name]
    high_predict_func = HIGH_MODEL_REGISTRY[high_model_name]

    if high_epochs is None:
        high_epochs = gru_epochs

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
        print(f"Low mode {idx + 1}: {low_model_name} len={len(preds)}")

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
        print(f"High mode {idx + 1}: {high_model_name} len={len(preds)}")

    high_pred_sum = np.sum(list(high_preds.values()), axis=0) if high_preds else np.zeros(len(test_data), dtype=float)

    if low_idx0 or high_idx0:
        target_modes = modes[low_idx0 + high_idx0]
    else:
        target_modes = modes
    modal_true_test = np.sum(target_modes, axis=0)[train_size:train_size + len(test_data)]

    final_pred = static_weight_fusion(low_pred, high_preds)
    final_pred = _to_float_1d(final_pred)
    _strict_len_check(final_pred, modal_true_test, len(test_data), "Fusion(static_weight_fusion)")

    final_pred = simple_residual_correction(modal_true_test, final_pred)

    true_sales = all_data[train_size:train_size + len(test_data)]
    true_sales_dates = all_dates[train_size:train_size + len(test_data)]
    _strict_len_check(final_pred, true_sales, len(test_data), "FinalPrediction")

    metrics = calculate_metrics(true_sales, final_pred, train_min, train_max) if len(true_sales) else {}
    if metrics:
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: NaN")

    for idx in low_idx0:
        mode_all = modes[idx]
        mode_true_all = np.concatenate([mode_all[:train_size], mode_all[train_size:train_size + len(test_data)]])
        mode_pred_padded = np.concatenate([np.full(train_size, np.nan), low_preds[idx]])
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
        mode_true_all = np.concatenate([mode_all[:train_size], mode_all[train_size:train_size + len(test_data)]])
        mode_pred_padded = np.concatenate([np.full(train_size, np.nan), high_preds[idx]])
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

    true_all = np.concatenate([all_data[:train_size], true_sales])
    final_pred_padded = np.concatenate([np.full(train_size, np.nan), final_pred])
    all_dates_plot = np.concatenate([all_dates[:train_size], true_sales_dates])

    plot_fusion_result(
        dates=all_dates_plot,
        y_true=true_all,
        y_pred=final_pred_padded,
        save_prefix=save_prefix,
        train_min=train_min,
        train_max=train_max,
    )

    result_df = pd.DataFrame({
        "Date": pd.to_datetime(true_sales_dates).strftime("%Y-%m-%d").tolist() if len(true_sales_dates) else [],
        "Actual_Sales": true_sales,
        "Final_Prediction": final_pred,
        "LowFreq_Sum_Prediction": low_pred,
        "HighFreq_Sum_Prediction": high_pred_sum,
        "Low_Model": low_model_name,
        "High_Model": high_model_name,
        "refit_interval": 0 if refit_interval is None else int(refit_interval),
    })

    for idx, arr in low_preds.items():
        result_df[f"Low_Mode_{idx + 1}_Pred"] = arr
    for idx, arr in high_preds.items():
        result_df[f"High_Mode_{idx + 1}_Pred"] = arr

    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"{save_prefix}_Prediction_Result.csv")
    result_df.to_csv(out_csv, index=False, float_format="%.6f", encoding="utf-8")
    print(f"Saved results: {os.path.abspath(out_csv)}")

    if len(true_sales) and len(final_pred):
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "grid.alpha": 0.3,
        })

        plt.figure(figsize=(6, 4))
        plt.plot(true_sales_dates, true_sales, label="True Value", linewidth=1.5)
        plt.plot(true_sales_dates, final_pred, label="Predicted Value", linestyle="--", linewidth=1.5)
        plt.title(f"{save_prefix} Sales Prediction ({low_model_name}+{high_model_name})", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10, frameon=True)
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{save_prefix}_Prediction_Result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {os.path.abspath(save_path)}")

        plt.figure(figsize=(6, 3))
        plt.plot(true_sales_dates, true_sales - final_pred, marker="o", linewidth=1.0, markersize=4)
        plt.axhline(y=0, linestyle="--", linewidth=1.0)
        plt.title(f"{save_prefix} Fusion Residuals", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        residual_dir = os.path.join("result", "Modal_Analysis", "Fusion_Prediction_Result")
        os.makedirs(residual_dir, exist_ok=True)
        save_path = os.path.join(residual_dir, f"{save_prefix}_Fusion_Residuals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved residuals: {os.path.abspath(save_path)}")

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
        raise ValueError("train_min/train_max must be provided")

    modes = np.asarray(modes, dtype=float)
    all_dates = np.asarray(all_dates)

    modes_train = modes[:, :train_size]
    modes_test = modes[:, train_size:]
    train_dates = all_dates[:train_size]
    test_dates = all_dates[train_size:]

    if len(test_dates) == 0:
        return pd.DataFrame()

    low_idx0 = [int(m) - 1 for m in low_modes_idx]

    if low_model_name not in LOW_MODEL_REGISTRY:
        raise ValueError(f"Unknown low_model_name={low_model_name}. Available: {list(LOW_MODEL_REGISTRY.keys())}")

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
        rows.append({
            "Mode_ID": idx + 1,
            "Low_Model": low_model_name,
            "refit_interval": 0 if refit_interval is None else int(refit_interval),
            **m,
        })

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
        raise ValueError("train_min/train_max must be provided")

    modes = np.asarray(modes, dtype=float)
    all_dates = np.asarray(all_dates)

    modes_train = modes[:, :train_size]
    modes_test = modes[:, train_size:]
    train_dates = all_dates[:train_size]
    test_dates = all_dates[train_size:]

    if len(test_dates) == 0:
        return pd.DataFrame()

    high_idx0 = [int(m) - 1 for m in high_modes_idx]

    if high_model_name not in HIGH_MODEL_REGISTRY:
        raise ValueError(f"Unknown high_model_name={high_model_name}. Available: {list(HIGH_MODEL_REGISTRY.keys())}")

    high_predict_func = HIGH_MODEL_REGISTRY[high_model_name]

    rows = []
    for idx in high_idx0:
        preds, y_true = _call_predictor(
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

        m = calculate_metrics(y_true, preds, train_min, train_max)
        rows.append({
            "Mode_ID": idx + 1,
            "High_Model": high_model_name,
            "refit_interval": 0 if refit_interval is None else int(refit_interval),
            **m,
        })

    return pd.DataFrame(rows)


vmd_prophet_gru_predict_optimized = vmd_xgboost_gru_fusion_forecast
