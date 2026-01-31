import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def setup_sci_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def load_and_split_data(file_name: str) -> Dict[str, Any]:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {os.path.abspath(file_name)}")

    df = pd.read_excel(file_name, engine="openpyxl", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    sales = pd.to_numeric(df["Sales"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"], errors="coerce").dt.to_pydatetime()

    mask = ~np.isnan(sales)
    sales = sales[mask]
    dates = np.asarray(dates)[mask]

    train_size = int(0.9 * len(sales))
    if len(sales) - train_size <= 5:
        train_size = int(0.8 * len(sales))
        print("Test set too small; train ratio set to 80%")

    train_data = sales[:train_size]
    test_data = sales[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    train_min = float(np.min(train_data)) if len(train_data) else np.nan
    train_max = float(np.max(train_data)) if len(train_data) else np.nan

    print(f"Split done: total={len(sales)} train={len(train_data)} test={len(test_data)}")
    print(f"Train range: min={train_min:.2f}, max={train_max:.2f}")

    return {
        "train_data": train_data,
        "test_data": test_data,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "all_dates": dates,
        "all_data": sales,
        "train_min": train_min,
        "train_max": train_max,
    }


def create_sequence_data(data: np.ndarray, seq_len: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    X, y = [], []
    for i in range(seq_len, len(data)):
        win = data[i - seq_len:i]
        target = data[i]
        if np.isnan(win).any() or np.isnan(target):
            continue
        X.append(win)
        y.append(target)
    if len(X) == 0:
        raise ValueError(f"Cannot create valid sequences (seq_len={seq_len})")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_min: float, train_max: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_ori = y_true[mask]
    y_pred_ori = y_pred[mask]

    keys = ["MAE", "MSE", "RMSE", "MAPE(%)", "SMAPE(%)", "DA(%)", "IA", "R", "R²"]
    if len(y_true_ori) == 0:
        return {k: np.nan for k in keys}

    denom = float(train_max - train_min)
    if not np.isfinite(denom) or abs(denom) < 1e-8:
        y_true_norm = np.zeros_like(y_true_ori)
        y_pred_norm = np.zeros_like(y_pred_ori)
    else:
        y_true_norm = (y_true_ori - train_min) / denom
        y_pred_norm = (y_pred_ori - train_min) / denom

    mae = mean_absolute_error(y_true_norm, y_pred_norm)
    mse = mean_squared_error(y_true_norm, y_pred_norm)
    rmse = float(np.sqrt(mse))

    eps = 1e-6
    mape = float(np.mean(np.abs((y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + eps))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + np.abs(y_pred_ori) + eps)) * 100.0)

    delta_true = y_true_ori[1:] - y_true_ori[:-1]
    delta_pred = y_pred_ori[1:] - y_pred_ori[:-1]
    mask_delta = delta_true != 0
    da = float(np.mean(np.sign(delta_true[mask_delta]) == np.sign(delta_pred[mask_delta])) * 100.0) if np.sum(mask_delta) > 0 else np.nan

    ia_num = float(np.sum((y_true_ori - y_pred_ori) ** 2))
    ia_den = float(np.sum((np.abs(y_pred_ori - np.mean(y_true_ori)) + np.abs(y_true_ori - np.mean(y_true_ori))) ** 2))
    ia = float(1.0 - (ia_num / (ia_den + eps)))

    r = float(pearsonr(y_true_ori, y_pred_ori)[0]) if len(y_true_ori) > 1 else np.nan
    r2 = float(r2_score(y_true_ori, y_pred_ori))

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "MAPE(%)": mape,
        "SMAPE(%)": smape,
        "DA(%)": da,
        "IA": ia,
        "R": r,
        "R²": r2,
    }


def _clean_filename(s: str) -> str:
    illegal = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '（', '）']
    for ch in illegal:
        s = s.replace(ch, "_")
    return s


def save_prediction_table(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    title_suffix: str,
    save_dir: str,
    category: str = "",
):
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(dates), len(y_true), len(y_pred))
    dates, y_true, y_pred = dates[:n], y_true[:n], y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    dates = dates[mask]
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    date_str = pd.to_datetime(dates).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "Date": date_str,
        "Actual_Sales": y_true,
        "Predicted_Sales": y_pred,
    })

    os.makedirs(save_dir, exist_ok=True)
    mn = _clean_filename(model_name)
    ts = _clean_filename(title_suffix)
    cat = _clean_filename(category) if category else ""
    file_name = f"{mn}_{cat}_Prediction({ts}).csv" if cat else f"{mn}_Prediction({ts}).csv"
    path = os.path.join(save_dir, file_name)

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {os.path.abspath(path)}")


def save_metrics_table(metrics_dict: Dict[str, float], model_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    df_new = pd.DataFrame([metrics_dict], index=[model_name])
    path = os.path.join(save_dir, "Prediction_Metrics_Summary.xlsx")

    if os.path.exists(path):
        df_exist = pd.read_excel(path, index_col=0)
        if model_name in df_exist.index:
            df_out = df_exist
        else:
            df_out = pd.concat([df_exist, df_new], axis=0)
    else:
        df_out = df_new

    df_out.to_excel(path, index=True)
    print(f"Saved metrics: {os.path.abspath(path)}")


def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    title_suffix: str = "",
    train_min: float = None,
    train_max: float = None,
    category: str = "",
):
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(dates), len(y_true), len(y_pred))
    dates, y_true, y_pred = dates[:n], y_true[:n], y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    dates_c = dates[mask]
    y_true_c = y_true[mask]
    y_pred_c = y_pred[mask]

    if len(y_true_c) == 0:
        print(f"{model_name}: no valid points to plot")
        return

    setup_sci_style()
    plt.figure(figsize=(6, 3), dpi=300)
    plt.plot(dates_c, y_true_c, label="True Value", linewidth=1.0)
    plt.plot(dates_c, y_pred_c, label="Predicted Value", linestyle="--", linewidth=1.0)

    cat = f" {category}" if category else ""
    suf = f" ({title_suffix})" if title_suffix else ""
    plt.title(f"{model_name}{cat} Sales Prediction{suf}", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Prediction_Result")
    os.makedirs(save_dir, exist_ok=True)

    mn = _clean_filename(model_name)
    ts = _clean_filename(title_suffix) if title_suffix else "Result"
    cat2 = _clean_filename(category) if category else ""
    file_name = f"{mn}_{cat2}_Prediction({ts}).png" if cat2 else f"{mn}_Prediction({ts}).png"
    path = os.path.join(save_dir, file_name)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved plot: {os.path.abspath(path)}")

    save_prediction_table(dates, y_true, y_pred, model_name, title_suffix or "Result", save_dir, category)

    if train_min is not None and train_max is not None:
        metrics = calculate_metrics(y_true, y_pred, train_min, train_max)
        save_metrics_table(metrics, model_name, save_dir)


def plot_fusion_result(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_prefix: str = "FDF",
    train_min: float = None,
    train_max: float = None,
):
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(dates), len(y_true), len(y_pred))
    dates, y_true, y_pred = dates[:n], y_true[:n], y_pred[:n]

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    dates_c = dates[mask]
    y_true_c = y_true[mask]
    y_pred_c = y_pred[mask]

    if len(y_true_c) == 0:
        print("Fusion: no valid points to plot")
        return

    setup_sci_style()
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(dates_c, y_true_c, label="True Value", linewidth=1.0)
    plt.plot(dates_c, y_pred_c, label="Predicted Value", linestyle="--", linewidth=1.0)

    plt.title(f"{save_prefix} Fusion Prediction", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    save_dir = os.path.join("result", "Modal_Analysis", "Fusion_Prediction_Result")
    os.makedirs(save_dir, exist_ok=True)

    sp = _clean_filename(save_prefix)
    path = os.path.join(save_dir, f"{sp}_Fusion_Prediction.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved fusion plot: {os.path.abspath(path)}")

    save_prediction_table(dates, y_true, y_pred, save_prefix, "Fusion", save_dir)

    if train_min is not None and train_max is not None:
        metrics = calculate_metrics(y_true, y_pred, train_min, train_max)
        save_metrics_table(metrics, save_prefix, save_dir)
