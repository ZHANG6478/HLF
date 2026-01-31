import os
import numpy as np
import pandas as pd
from vmdpy.vmdpy import VMD
from typing import Optional

from common import load_and_split_data
from vmd import optimize_vmd_params, analyze_vmd_modes, split_imfs_by_frequency_and_energy
from predict import evaluate_low_modes_metrics


CATEGORY_FILES = [
    ("Tops", "Tops_timeseries.xlsx"),
    ("Others", "Others_timeseries.xlsx"),
    ("Outerwear", "Outerwear_timeseries.xlsx"),
    ("Bags", "Bags_timeseries.xlsx"),
    ("Skirts", "Skirts_timeseries.xlsx"),
    ("Pants", "Pants_timeseries.xlsx"),
    ("Jumpsuits", "Jumpsuits_timeseries.xlsx"),
    ("Accessories", "Accessories_timeseries.xlsx"),
    ("Footwear", "Footwear_timeseries.xlsx"),
]


def run_one_category_low(
    category: str,
    data_file: str,
    low_model_name: str = "XGBoost",
    energy_thresh: float = 0.90,
    seq_len: int = 7,
    low_epochs: Optional[int] = None,
    refit_interval: Optional[int] = 7,
    k_min: int = 2,
    k_max: int = 10,
    alpha_min: int = 500,
    alpha_max: int = 1500,
    alpha_step: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    data_dict = load_and_split_data(data_file)
    train_min = data_dict["train_min"]
    train_max = data_dict["train_max"]
    all_data, all_dates = data_dict["all_data"], data_dict["all_dates"]
    train_size = len(data_dict["train_data"])

    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    if len(test_data) == 0:
        print(f"[{category}] No test data, skip.")
        return pd.DataFrame()

    vmd_opt = optimize_vmd_params(
        data=train_data,
        K_range=range(k_min, k_max + 1),
        alpha_range=range(alpha_min, alpha_max + 1, alpha_step),
    )
    K_opt, alpha_opt = int(vmd_opt["K"]), int(vmd_opt["alpha"])
    print(f"[{category}] VMD optimized: K={K_opt}, alpha={alpha_opt}")

    modes, omega, _ = VMD(all_data, alpha_opt, 0, K_opt, 0, 1, 1e-6)
    modes = np.array(modes) if isinstance(modes, list) else np.asarray(modes, dtype=float)

    df_mode = analyze_vmd_modes(modes, omega)
    low_modes, high_modes, df_sorted = split_imfs_by_frequency_and_energy(df_mode, energy_thresh)
    print(f"[{category}] Low modes={low_modes} | High modes={high_modes}")

    if not low_modes:
        print(f"[{category}] No low-frequency modes under threshold={energy_thresh}, skip.")
        return pd.DataFrame()

    df_low_metrics = evaluate_low_modes_metrics(
        modes=modes,
        all_dates=all_dates,
        train_size=train_size,
        low_modes_idx=low_modes,
        low_model_name=low_model_name,
        seq_len=seq_len,
        low_epochs=low_epochs,
        refit_interval=refit_interval,
        train_min=train_min,
        train_max=train_max,
        verbose=verbose,
    )

    if df_low_metrics.empty:
        return df_low_metrics

    df_low_info = df_sorted[df_sorted["Mode_ID"].isin(low_modes)][
        ["Mode_ID", "f_mean (Hz)", "Energy_Ratio (%)", "Cumulative_Energy"]
    ].copy()

    out = df_low_metrics.merge(df_low_info, on="Mode_ID", how="left")
    out.insert(0, "Category", category)
    out.insert(1, "Data_File", data_file)
    out.insert(2, "Energy_Thresh", energy_thresh)

    save_dir = os.path.join("result", "Low_Model_Selection")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{category}_LowModes_{low_model_name}_Metrics.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[{category}] Saved: {os.path.abspath(out_path)}")

    return out


def main():
    low_model = "DLinear"
    energy = 0.90
    seq_len = 7
    refit = 0
    low_epochs = 50

    all_rows = []
    for category, file_name in CATEGORY_FILES:
        df = run_one_category_low(
            category=category,
            data_file=file_name,
            low_model_name=low_model,
            energy_thresh=energy,
            seq_len=seq_len,
            low_epochs=low_epochs,
            refit_interval=refit,
            verbose=False,
        )
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No outputs generated.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    save_dir = os.path.join("result", "Low_Model_Selection")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"ALL_Categories_LowModes_{low_model}_Metrics.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[ALL] Saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
