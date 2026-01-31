import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


def setup_chinese_display():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (14, 10)


def load_and_split_data(file_name):
    if not os.path.exists(file_name):
        print(f"Error: file not found: {os.path.abspath(file_name)}")
        exit()

    df = pd.read_excel(io=file_name, engine="openpyxl", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    mask_valid = ~df["Sales"].isna()
    if (~mask_valid).any():
        print(f"Warning: dropped {(~mask_valid).sum()} NaN rows from Sales")
    df = df.loc[mask_valid].reset_index(drop=True)

    smoothed_sales = df["Sales"].values.astype(float)
    dates = df["date"].dt.to_pydatetime()

    train_size = int(0.9 * len(smoothed_sales))
    if len(smoothed_sales) - train_size <= 5:
        train_size = int(0.8 * len(smoothed_sales))
        print("Warning: test set too small, train ratio adjusted to 0.8")

    train_data = smoothed_sales[:train_size]
    test_data = smoothed_sales[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    train_min = np.min(train_data)
    train_max = np.max(train_data)

    print("Split done:")
    print(f"  total: {len(smoothed_sales)}")
    print(f"  train: {len(train_data)} ({train_dates[0].strftime('%Y-%m-%d')} .. {train_dates[-1].strftime('%Y-%m-%d')})")
    print(f"  test : {len(test_data)} ({test_dates[0].strftime('%Y-%m-%d')} .. {test_dates[-1].strftime('%Y-%m-%d')})")
    print(f"  train range: min={train_min:.2f}, max={train_max:.2f}")

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


def create_sequence_data(data, seq_len=7):
    data = np.asarray(data, dtype=float)
    X, y = [], []
    for i in range(seq_len, len(data)):
        if np.isnan(data[i - seq_len:i]).any() or np.isnan(data[i]):
            continue
        X.append(data[i - seq_len:i])
        y.append(data[i])
    if len(X) == 0:
        raise ValueError(f"Cannot create sequences (seq_len={seq_len})")
    return np.array(X), np.array(y)


def normalize_for_metrics(y_true, y_pred, train_min, train_max):
    if train_max - train_min < 1e-8:
        return np.zeros_like(y_true), np.zeros_like(y_pred)

    y_true_norm = np.where(np.isnan(y_true), np.nan, (y_true - train_min) / (train_max - train_min))
    y_pred_norm = np.where(np.isnan(y_pred), np.nan, (y_pred - train_min) / (train_max - train_min))
    return y_true_norm, y_pred_norm


def calculate_metrics(y_true, y_pred, train_min, train_max):
    mask_original = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_ori = y_true[mask_original]
    y_pred_ori = y_pred[mask_original]

    if len(y_true_ori) == 0:
        print("Warning: no valid data for metrics")
        return {k: np.nan for k in ["MAE", "MSE", "RMSE", "MAPE(%)", "SMAPE(%)", "DA(%)", "IA", "R", "R²"]}

    y_true_norm, y_pred_norm = normalize_for_metrics(y_true, y_pred, train_min, train_max)
    y_true_norm_clean = y_true_norm[mask_original]
    y_pred_norm_clean = y_pred_norm[mask_original]

    mae = mean_absolute_error(y_true_norm_clean, y_pred_norm_clean)
    mse = mean_squared_error(y_true_norm_clean, y_pred_norm_clean)
    rmse = np.sqrt(mse)

    eps = 1e-6
    mape = np.mean(np.abs((y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + eps))) * 100
    smape = np.mean(2 * np.abs(y_true_ori - y_pred_ori) / (np.abs(y_true_ori) + np.abs(y_pred_ori) + eps)) * 100

    delta_true = y_true_ori[1:] - y_true_ori[:-1]
    delta_pred = y_pred_ori[1:] - y_pred_ori[:-1]
    mask_delta = delta_true != 0
    da = np.mean(np.sign(delta_true[mask_delta]) == np.sign(delta_pred[mask_delta])) * 100 if np.sum(mask_delta) > 0 else np.nan

    ia_num = np.sum((y_true_ori - y_pred_ori) ** 2)
    ia_den = np.sum((np.abs(y_pred_ori - np.mean(y_true_ori)) + np.abs(y_true_ori - np.mean(y_true_ori))) ** 2)
    ia = 1 - (ia_num / (ia_den + eps))

    r, _ = pearsonr(y_true_ori, y_pred_ori)
    r2 = r2_score(y_true_ori, y_pred_ori)

    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE(%)": mape, "SMAPE(%)": smape, "DA(%)": da,
        "IA": ia, "R": r, "R²": r2,
    }


def plot_predictions(dates, y_true, y_pred, model_name, title_suffix=""):
    n = min(len(dates), len(y_true), len(y_pred))
    dates = np.asarray(dates[:n])
    y_true = np.asarray(y_true[:n], dtype=float)
    y_pred = np.asarray(y_pred[:n], dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    dates_clean = dates[mask]
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print(f"Warning: {model_name}: no valid data to plot")
        return

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(dates_clean, y_true_clean, label="Actual", linewidth=1.5)
    ax.plot(dates_clean, y_pred_clean, label="Predicted", linewidth=1.5)

    ax.set_title(f"{title_suffix} Sales Prediction ({model_name})", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sales", fontsize=12)

    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(linestyle="--", alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()

    os.makedirs("result", exist_ok=True)
    save_path = os.path.join("result", f"{title_suffix}_Prediction({model_name}).png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {os.path.abspath(save_path)}")

    plt.rcParams.update(plt.rcParamsDefault)

    save_predictions_to_csv(dates_clean, y_true_clean, y_pred_clean, model_name, title_suffix)


def save_predictions_to_csv(dates, y_true, y_pred, model_name, title_suffix=""):
    mask = ~np.isnan(y_true)
    dates_clean = np.asarray(dates)[mask]
    y_true_clean = np.asarray(y_true)[mask]
    y_pred_clean = np.asarray(y_pred)[mask]

    date_str = pd.to_datetime(dates_clean).strftime("%Y-%m-%d")

    df_result = pd.DataFrame({
        "Date": date_str,
        "Actual Sales": y_true_clean,
        "Predicted Sales": y_pred_clean,
    })

    save_path = os.path.join("result", f"{title_suffix}_Prediction({model_name}).csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_result.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {os.path.abspath(save_path)}")


def residual_correction(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if len(y_true) < 3:
        return y_pred
    residual = y_true - y_pred
    max_correction = 0.00 * np.abs(y_pred)
    correction = np.clip(residual, -max_correction, max_correction)
    return y_pred + correction
