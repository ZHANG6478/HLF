import os
import numpy as np
import pandas as pd
import torch
from vmdpy.vmdpy import VMD
from typing import Tuple, Dict

from common import load_and_split_data
from utils import plot_vmd_modes, plot_mode_reconstruction
from vmd import optimize_vmd_params, analyze_vmd_modes, split_imfs_by_frequency_and_energy
from predict import vmd_prophet_gru_predict_optimized


def vmd_xgboost_gru_forecast(
    data_file: str = "Tops_timeseries.xlsx",
    k_min: int = 2,
    k_max: int = 10,
    alpha_min: int = 500,
    alpha_max: int = 1500,
    alpha_step: int = 100,
    energy_thresh: float = 0.90,
    seq_len: int = 7,
    epochs: int = 150,
    refit_interval: int = 7,
    low_model_name: str = "XGBoost",
    high_model_name: str = "GRU",
    save_prefix: str = "Tops",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = load_and_split_data(data_file)

    all_data = data["all_data"]
    all_dates = data["all_dates"]
    train_size_ori = len(data["train_data"])
    train_min, train_max = data["train_min"], data["train_max"]

    train_data = all_data[:train_size_ori]
    test_data = all_data[train_size_ori:]
    train_dates = all_dates[:train_size_ori]
    test_dates = all_dates[train_size_ori:]

    concat_data = np.concatenate([train_data, test_data])
    concat_dates = np.concatenate([train_dates, test_dates])

    train_size = min(train_size_ori, len(concat_data))
    train_data = concat_data[:train_size]
    test_data = concat_data[train_size:]
    train_dates = concat_dates[:train_size]
    test_dates = concat_dates[train_size:]
    all_data, all_dates = concat_data, concat_dates

    print(f"Data: total={len(all_data)} train={len(train_data)} test={len(test_data)}")
    if len(test_data) == 0:
        return pd.DataFrame(), {}

    vmd_opt = optimize_vmd_params(
        data=train_data,
        K_range=range(k_min, k_max + 1),
        alpha_range=range(alpha_min, alpha_max + 1, alpha_step),
    )
    K_opt, alpha_opt = vmd_opt["K"], vmd_opt["alpha"]
    print(f"VMD params: K={K_opt} alpha={alpha_opt}")

    modes, omega, _ = VMD(all_data, alpha_opt, 0, K_opt, 0, 1, 1e-6)
    modes = np.asarray(modes)
    print(f"VMD done: modes={modes.shape}")

    plot_vmd_modes(modes, all_dates, save_prefix="VMD")
    plot_mode_reconstruction(all_data, modes, all_dates, save_prefix="VMD")

    df_mode = analyze_vmd_modes(modes, omega)
    low_modes, high_modes, df_sorted = split_imfs_by_frequency_and_energy(df_mode, energy_thresh)

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)
    df_sorted.to_csv(os.path.join(save_dir, "Modal_Features.csv"), index=False, encoding="utf-8")
    print(f"Split: low={low_modes} high={high_modes}")

    result_df, metrics = vmd_prophet_gru_predict_optimized(
        modes=modes,
        omega=omega,
        all_dates=all_dates,
        all_data=all_data,
        train_size=train_size,
        low_modes_idx=low_modes,
        high_modes_idx=high_modes,
        seq_len=seq_len,
        gru_epochs=epochs,
        refit_interval=refit_interval,
        low_model_name=low_model_name,
        high_model_name=high_model_name,
        energy_threshold=energy_thresh,
        save_prefix=save_prefix,
        train_min=train_min,
        train_max=train_max,
        verbose=verbose,
    )

    print("Done")
    return result_df, metrics


if __name__ == "__main__":
    pred_result, pred_metrics = vmd_xgboost_gru_forecast(
        data_file="Tops_timeseries.xlsx",
        energy_thresh=0.90,
        seq_len=7,
        epochs=100,
        refit_interval=0,
        low_model_name="DLinear",
        high_model_name="NBEATS",
        save_prefix="Tops",
        verbose=False,
    )

    print("\nMetrics")
    for k, v in pred_metrics.items():
        print(f"{k}: {v:.4f}" if not np.isnan(v) else f"{k}: NaN")
