import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def load_exchange_series(file_name: str, value_col: str) -> Dict[str, Any]:
    if not os.path.exists(file_name):
        raise FileNotFoundError(file_name)

    df = pd.read_csv(file_name, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    values = df[str(value_col)].astype(float).values
    dates = df["date"].dt.to_pydatetime()

    train_size = int(0.9 * len(values))
    train_data = values[:train_size]
    test_data = values[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    train_min = float(np.min(train_data))
    train_max = float(np.max(train_data))

    return {
        "train_data": train_data,
        "test_data": test_data,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "train_min": train_min,
        "train_max": train_max,
    }


def setup_sci_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )


def load_and_split_data(file_name: str) -> Dict[str, Any]:
    if not os.path.exists(file_name):
        raise FileNotFoundError(file_name)

    df = pd.read_excel(file_name, engine="openpyxl", parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)

    smoothed_sales = df["5天平滑"].values
    mask = ~np.isnan(smoothed_sales)
    smoothed_sales = smoothed_sales[mask]
    dates = df["日期"].dt.to_pydatetime()[mask]

    train_size = int(0.9 * len(smoothed_sales))
    if len(smoothed_sales) - train_size <= 5:
        train_size = int(0.8 * len(smoothed_sales))

    train_data = smoothed_sales[:train_size]
    test_data = smoothed_sales[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    train_min = float(np.min(train_data))
    train_max = float(np.max(train_data))

    return {
        "train_data": train_data,
        "test_data": test_data,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "all_dates": dates,
        "all_data": smoothed_sales,
        "train_min": train_min,
        "train_max": train_max,
    }


def create_sequence_data(data: np.ndarray, seq_len: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(int(seq_len), len(data)):
        win = data[i - int(seq_len) : i]
        target = data[i]
        if not np.isnan(win).any() and not np.isnan(target):
            X.append(win)
            y.append(target)

    if len(X) == 0:
        raise ValueError(f"Invalid sequence generation (seq_len={seq_len})")

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_min: float, train_max: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_ori = y_true[mask]
    y_pred_ori = y_pred[mask]

    keys = ["MAE", "MSE", "RMSE", "MAPE(%)", "SMAPE(%)", "DA(%)", "IA", "R", "R2"]
    if len(y_true_ori) == 0:
        return {k: np.nan for k in keys}

    denom = float(train_max - train_min)
    if abs(denom) < 1e-8:
        y_true_norm = np.zeros_like(y_true_ori)
        y_pred_norm = np.zeros_like(y_pred_ori)
    else:
        y_true_norm = (y_true_ori - train_min) / denom
        y_pred_norm = (y_pred_ori - train_min) / denom

    mae = float(mean_absolute_error(y_true_norm, y_pred_norm))
    mse = float(mean_squared_error(y_true_norm, y_pred_norm))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_ori, y_pred_ori))

    eps = 1e-6
    mape = float(np.mean(np.abs((y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + eps))) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + np.abs(y_pred_ori) + eps)) * 100.0
    )

    if len(y_true_ori) >= 2:
        delta_true = y_true_ori[1:] - y_true_ori[:-1]
        delta_pred = y_pred_ori[1:] - y_pred_ori[:-1]
        mask_delta = delta_true != 0
        if np.sum(mask_delta) > 0:
            da = float(np.mean(np.sign(delta_true[mask_delta]) == np.sign(delta_pred[mask_delta])) * 100.0)
        else:
            da = np.nan
    else:
        da = np.nan

    ia_num = float(np.sum((y_true_ori - y_pred_ori) ** 2))
    ia_den = float(
        np.sum((np.abs(y_pred_ori - np.mean(y_true_ori)) + np.abs(y_true_ori - np.mean(y_true_ori))) ** 2)
    )
    ia = float(1.0 - (ia_num / (ia_den + eps)))

    if len(y_true_ori) >= 2:
        r, _ = pearsonr(y_true_ori, y_pred_ori)
        r = float(r)
    else:
        r = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "SMAPE(%)": smape,
        "DA(%)": da,
        "IA": ia,
        "R": r,
        "R2": r2,
    }


def save_prediction_table(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    title_suffix: str,
    save_dir: str,
    category: str = "",
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true)
    dates_clean = np.asarray(dates)[mask]
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    df = pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in dates_clean],
            "Actual_Sales": y_true_clean,
            "Predicted_Sales": y_pred_clean,
        }
    )

    def clean_filename(s: str) -> str:
        illegal_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        for ch in illegal_chars:
            s = s.replace(ch, "_")
        return s

    model_name_clean = clean_filename(model_name)
    title_suffix_clean = clean_filename(title_suffix)

    if category:
        file_name = f"{model_name_clean}_{category}_Prediction({title_suffix_clean}).csv"
    else:
        file_name = f"{model_name_clean}_Prediction({title_suffix_clean}).csv"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def save_metrics_table(metrics_dict: Dict[str, float], model_name: str, save_dir: str) -> None:
    df = pd.DataFrame([metrics_dict], index=[model_name])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Prediction_Metrics_Summary.xlsx")

    if os.path.exists(save_path):
        df_exist = pd.read_excel(save_path, index_col=0)
        if model_name not in df_exist.index:
            df = pd.concat([df_exist, df])
        else:
            df = df_exist

    df.to_excel(save_path, index=True)


def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    title_suffix: str = "",
    train_min: float = None,
    train_max: float = None,
    category: str = "",
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true)
    dates_clean = np.asarray(dates)[mask]
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    setup_sci_style()
    plt.figure(figsize=(6, 3), dpi=300)

    plt.plot(dates_clean, y_true_clean, label="True Value", color=(0.5, 0.5, 1.0), linewidth=1.0)
    plt.plot(dates_clean, y_pred_clean, label="Predicted Value", color=(1.0, 0.5, 0.5), linestyle="--", linewidth=1.0)

    plt.title(f"{model_name} {category} Sales Prediction ({title_suffix})", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Prediction_Result")
    os.makedirs(save_dir, exist_ok=True)

    def clean_filename(s: str) -> str:
        illegal_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", "（", "）"]
        for ch in illegal_chars:
            s = s.replace(ch, "_")
        return s

    model_name_clean = clean_filename(model_name)
    title_suffix_clean = clean_filename(title_suffix)

    file_name = f"{model_name_clean}_{category}_Prediction({title_suffix_clean}).png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    save_prediction_table(dates, y_true, y_pred, model_name, title_suffix, save_dir, category)

    if train_min is not None and train_max is not None:
        metrics = calculate_metrics(y_true, y_pred, train_min, train_max)
        save_metrics_table(metrics, model_name, save_dir)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.close()


def plot_fusion_result(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_prefix: str = "VXG",
    train_min: float = None,
    train_max: float = None,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true)
    dates_clean = np.asarray(dates)[mask]
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    setup_sci_style()
    plt.figure(figsize=(8, 4), dpi=300)

    plt.plot(dates_clean, y_true_clean, label="True Value", color=(0.5, 0.5, 1.0), linewidth=1.0)
    plt.plot(dates_clean, y_pred_clean, label="Predicted Value", color=(1.0, 0.5, 0.5), linestyle="--", linewidth=1.0)

    plt.title(f"{save_prefix} Fusion Prediction", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    save_dir = os.path.join("result", "Modal_Analysis", "Fusion_Prediction_Result")
    os.makedirs(save_dir, exist_ok=True)

    save_prefix_clean = save_prefix.replace("/", "_").replace("\\", "_")
    file_name = f"{save_prefix_clean}_Fusion_Prediction.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    save_prediction_table(dates, y_true, y_pred, save_prefix, "Fusion", save_dir)

    if train_min is not None and train_max is not None:
        metrics = calculate_metrics(y_true, y_pred, train_min, train_max)
        save_metrics_table(metrics, save_prefix, save_dir)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.close()
