# main.py
import os
import random
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from vmdpy.vmdpy import VMD

from common import load_exchange_series, calculate_metrics
from utils import plot_vmd_modes, plot_mode_reconstruction
from vmd import analyze_vmd_modes, split_imfs_by_frequency_and_energy
from predict import vmd_xgboost_gru_fusion_forecast


SEED = 37
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    import tensorflow as tf

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def vmd_xgboost_gru_forecast_exchange(
    data_file: str = "exchange_rate.csv",
    value_col: str = "3",
    k_min: int = 2,
    k_max: int = 10,
    alpha_min: int = 500,
    alpha_max: int = 1500,
    alpha_step: int = 100,
    energy_thresh: float = 0.90,
    seq_len: int = 96,
    epochs: int = 50,
    refit_interval=None,
    low_model_name: str = "XGBoost",
    high_model_name: str = "GRU",
    save_prefix: str = "Exchange",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = load_exchange_series(data_file, value_col)
    train_data = data["train_data"]
    test_data = data["test_data"]
    train_dates = data["train_dates"]
    test_dates = data["test_dates"]
    train_min = data["train_min"]
    train_max = data["train_max"]

    all_data = np.concatenate([train_data, test_data])
    all_dates = np.concatenate([train_dates, test_dates])
    train_size = len(train_data)

    if len(test_data) == 0:
        return pd.DataFrame(), {}

    K_opt, alpha_opt = 10, 500

    modes, omega, _ = VMD(all_data, alpha_opt, 0, K_opt, 0, 1, 1e-6)
    modes = np.array(modes) if isinstance(modes, list) else modes

    plot_vmd_modes(modes, all_dates)
    plot_mode_reconstruction(all_data, modes, all_dates)

    df_mode = analyze_vmd_modes(modes, omega)
    low_modes, high_modes, df_sorted = split_imfs_by_frequency_and_energy(df_mode, energy_thresh)

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)
    df_sorted.to_csv(
        os.path.join(save_dir, f"Exchange_{value_col}_Modal_Features.csv"),
        index=False,
        encoding="utf-8",
    )

    result_df, metrics = vmd_xgboost_gru_fusion_forecast(
        modes=modes,
        omega=omega,
        all_dates=all_dates,
        all_data=all_data,
        train_size=train_size,
        low_modes_idx=low_modes,
        high_modes_idx=high_modes,
        seq_len=seq_len,
        low_epochs=epochs,
        high_epochs=epochs,
        refit_interval=refit_interval,
        low_model_name=low_model_name,
        high_model_name=high_model_name,
        energy_threshold=energy_thresh,
        save_prefix=f"{save_prefix}_col{value_col}",
        train_min=train_min,
        train_max=train_max,
        verbose=verbose,
    )

    if (
        result_df is not None
        and len(result_df) > 0
        and ("y_true" in result_df.columns)
        and ("y_pred" in result_df.columns)
    ):
        metrics = calculate_metrics(
            result_df["y_true"].values,
            result_df["y_pred"].values,
            train_min,
            train_max,
        )

    return result_df, metrics


if __name__ == "__main__":
    pred_result, pred_metrics = vmd_xgboost_gru_forecast_exchange(
        data_file="exchange_rate.csv",
        value_col="3",
        energy_thresh=0.9999,
        seq_len=96,
        epochs=100,
        refit_interval=None,
        low_model_name="XGBoost",
        high_model_name="GRU",
        save_prefix="Exchange",
        verbose=False,
    )

    for k, v in pred_metrics.items():
        if np.isnan(v):
            print(f"{k}: NaN")
        else:
            print(f"{k}: {v:.6f}")
