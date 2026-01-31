import numpy as np
import pandas as pd
from vmdpy.vmdpy import VMD
from scipy.signal import hilbert
from typing import Tuple, List
from utils import envelope_entropy


def optimize_vmd_params(
    data: np.ndarray,
    K_range: range,
    alpha_range: list,
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
                    "alpha": float(alpha),
                    "mse": float(mse),
                    "entropy": float(avg_env_entropy),
                    "fitness": 0.0,
                }
            )

    all_mse = np.asarray([r["mse"] for r in results], dtype=float)
    all_entropy = np.asarray([r["entropy"] for r in results], dtype=float)

    mse_min, mse_max = float(np.min(all_mse)), float(np.max(all_mse))
    entropy_min, entropy_max = float(np.min(all_entropy)), float(np.max(all_entropy))

    mse_range = (mse_max - mse_min) if mse_max != mse_min else 1e-8
    entropy_range = (entropy_max - entropy_min) if entropy_max != entropy_min else 1e-8

    for r in results:
        norm_mse = (mse_max - r["mse"]) / mse_range
        norm_entropy = (entropy_max - r["entropy"]) / entropy_range
        r["fitness"] = float(0.5 * norm_entropy + 0.5 * norm_mse)

    mse_threshold = float(np.percentile(all_mse, 20))
    qualified = [r for r in results if r["mse"] <= mse_threshold]
    best_res = max(qualified, key=lambda x: x["fitness"])

    return best_res


def analyze_imf_frequency(imf: np.ndarray, Fs: float = 1.0) -> Tuple[float, float, float, float]:
    n = len(imf)

    fft_vals = np.fft.fft(imf)
    freqs = np.fft.fftfreq(n, d=1 / Fs)
    pos_freq = freqs > 0
    f_peak = float(freqs[pos_freq][np.argmax(np.abs(fft_vals[pos_freq]))]) if np.any(pos_freq) else 0.0

    analytic_sig = hilbert(imf)
    inst_phase = np.unwrap(np.angle(analytic_sig))
    inst_freq = np.diff(inst_phase) * Fs / (2 * np.pi)
    inst_freq = inst_freq[inst_freq > 0]
    f_mean = float(np.mean(inst_freq)) if len(inst_freq) > 0 else 0.0

    T_peak = float(1 / f_peak) if f_peak > 0 else float(np.inf)
    T_mean = float(1 / f_mean) if f_mean > 0 else float(np.inf)

    return f_peak, f_mean, T_peak, T_mean


def analyze_vmd_modes(u: np.ndarray, omega: np.ndarray) -> pd.DataFrame:
    n_modes = u.shape[0]
    total_energy = float(np.sum([np.sum(mode**2) for mode in u]))
    mode_features = []

    for k in range(n_modes):
        mode = u[k]
        energy = float(np.sum(mode**2))
        energy_ratio = float(energy / total_energy * 100.0) if total_energy != 0 else 0.0

        f_peak, f_mean, T_peak, T_mean = analyze_imf_frequency(mode)

        omega_k = omega[k]
        omega_center = float(np.real(omega_k[-1])) if np.ndim(omega_k) > 0 else float(np.real(omega_k))
        f_center = float(omega_center / (2 * np.pi))

        mode_features.append(
            {
                "Mode_ID": int(k + 1),
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

    return pd.DataFrame(mode_features)


def split_imfs_by_frequency_and_energy(
    df_mode_info: pd.DataFrame, energy_threshold: float = 0.9
) -> Tuple[List[int], List[int], pd.DataFrame]:
    df_sorted = df_mode_info.sort_values("f_mean (Hz)").reset_index(drop=True)
    df_sorted["Cumulative_Energy"] = df_sorted["Energy_Ratio (%)"].cumsum() / 100.0

    low_modes = df_sorted[df_sorted["Cumulative_Energy"] <= energy_threshold]["Mode_ID"].tolist()
    high_modes = df_sorted[df_sorted["Cumulative_Energy"] > energy_threshold]["Mode_ID"].tolist()

    return low_modes, high_modes, df_sorted
