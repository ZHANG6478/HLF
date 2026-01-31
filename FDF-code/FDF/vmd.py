import numpy as np
import pandas as pd
from vmdpy.vmdpy import VMD
from scipy.signal import hilbert
from typing import Tuple, List
from utils import envelope_entropy


def optimize_vmd_params(
    data: np.ndarray,
    K_range: range,
    alpha_range: range,
) -> dict:
    results = []

    for K in K_range:
        for alpha in alpha_range:
            u, _, _ = VMD(data, alpha, 0, K, 0, 1, 1e-6)

            reconstructed = np.sum(u, axis=0)
            mse = np.mean((data[: len(reconstructed)] - reconstructed) ** 2)
            avg_env_entropy = float(np.mean([envelope_entropy(imf) for imf in u]))

            results.append(
                {
                    "K": int(K),
                    "alpha": int(alpha),
                    "mse": float(mse),
                    "entropy": avg_env_entropy,
                    "fitness": 0.0,
                }
            )

    all_mse = np.array([r["mse"] for r in results], dtype=float)
    all_entropy = np.array([r["entropy"] for r in results], dtype=float)

    mse_min, mse_max = float(np.min(all_mse)), float(np.max(all_mse))
    ent_min, ent_max = float(np.min(all_entropy)), float(np.max(all_entropy))
    mse_range = (mse_max - mse_min) if mse_max != mse_min else 1e-8
    ent_range = (ent_max - ent_min) if ent_max != ent_min else 1e-8

    for r in results:
        norm_mse = (mse_max - r["mse"]) / mse_range
        norm_ent = (ent_max - r["entropy"]) / ent_range
        r["fitness"] = float(0.5 * norm_ent + 0.5 * norm_mse)

    mse_threshold = float(np.percentile(all_mse, 20))
    qualified = [r for r in results if r["mse"] <= mse_threshold]
    best_res = max(qualified, key=lambda x: x["fitness"])

    print(
        f"Best VMD params: K={best_res['K']}, alpha={best_res['alpha']} | "
        f"fitness={best_res['fitness']:.6f}, mse={best_res['mse']:.6f}, entropy={best_res['entropy']:.6f}"
    )
    return best_res


def analyze_imf_frequency(imf: np.ndarray, Fs: float = 1.0) -> Tuple[float, float, float, float]:
    n = int(len(imf))

    fft_vals = np.fft.fft(imf)
    freqs = np.fft.fftfreq(n, d=1.0 / Fs)
    pos = freqs > 0
    if np.any(pos):
        f_peak = float(freqs[pos][np.argmax(np.abs(fft_vals[pos]))])
    else:
        f_peak = 0.0

    analytic_sig = hilbert(imf)
    inst_phase = np.unwrap(np.angle(analytic_sig))
    inst_freq = np.diff(inst_phase) * Fs / (2.0 * np.pi)
    inst_freq = inst_freq[inst_freq > 0]
    f_mean = float(np.mean(inst_freq)) if inst_freq.size > 0 else 0.0

    T_peak = (1.0 / f_peak) if f_peak > 0 else float("inf")
    T_mean = (1.0 / f_mean) if f_mean > 0 else float("inf")
    return f_peak, f_mean, T_peak, T_mean


def analyze_vmd_modes(u: np.ndarray, omega: np.ndarray) -> pd.DataFrame:
    u = np.asarray(u, dtype=float)
    n_modes = int(u.shape[0])
    total_energy = float(np.sum(u ** 2))

    rows = []
    for k in range(n_modes):
        mode = u[k]
        energy = float(np.sum(mode ** 2))
        energy_ratio = (energy / total_energy * 100.0) if total_energy > 0 else 0.0

        f_peak, f_mean, T_peak, T_mean = analyze_imf_frequency(mode)

        omega_k = omega[k]
        omega_center = float(np.real(omega_k[-1])) if np.ndim(omega_k) > 0 else float(np.real(omega_k))
        f_center = float(omega_center / (2.0 * np.pi))

        rows.append(
            {
                "Mode_ID": k + 1,
                "omega_center (rad/sample)": omega_center,
                "f_center (Hz)": f_center,
                "f_peak (Hz)": f_peak,
                "f_mean (Hz)": f_mean,
                "T_peak (s)": T_peak,
                "T_mean (s)": T_mean,
                "Energy": energy,
                "Energy_Ratio (%)": energy_ratio,
            }
        )

    return pd.DataFrame(rows)


def split_imfs_by_frequency_and_energy(
    df_mode_info: pd.DataFrame,
    energy_threshold: float = 0.9,
) -> Tuple[List[int], List[int], pd.DataFrame]:
    df_sorted = df_mode_info.sort_values("f_mean (Hz)").reset_index(drop=True)
    df_sorted["Cumulative_Energy"] = df_sorted["Energy_Ratio (%)"].cumsum() / 100.0

    low_modes = df_sorted[df_sorted["Cumulative_Energy"] <= energy_threshold]["Mode_ID"].astype(int).tolist()
    high_modes = df_sorted[df_sorted["Cumulative_Energy"] > energy_threshold]["Mode_ID"].astype(int).tolist()

    return low_modes, high_modes, df_sorted
