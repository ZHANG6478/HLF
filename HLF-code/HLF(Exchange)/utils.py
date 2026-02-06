import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def envelope_entropy(data: np.ndarray) -> float:
    from scipy.signal import hilbert

    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    s = float(np.sum(amplitude_envelope))
    if s <= 0:
        return 0.0
    p = amplitude_envelope / s
    return float(-np.sum(p * np.log(p + 1e-12)))


def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    n = len(data)
    std_data = float(np.std(data))
    tol = float(r * std_data) if std_data > 0 else 1e-6

    def _phi(m_len: int) -> float:
        x = np.array([data[i : i + m_len] for i in range(n - m_len)])
        diff = np.max(np.abs(x[:, np.newaxis] - x), axis=2)
        count = float(np.sum(diff <= tol) - (n - m_len))
        denom = float((n - m_len) * (n - m_len - 1))
        return count / denom if denom > 0 else 0.0

    phi_m = _phi(int(m))
    phi_m1 = _phi(int(m) + 1)
    if phi_m == 0.0 or phi_m1 == 0.0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def plot_vmd_modes(modes: np.ndarray, dates: np.ndarray, save_prefix: str = "VMD"):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )

    n_modes = int(modes.shape[0])
    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_modes):
        plt.figure(figsize=(6, 2), dpi=300)
        plt.plot(dates, modes[i], color=f"C{i}", linewidth=1.0)
        plt.title(f"Decomposition Mode {i + 1}", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(linestyle="--", alpha=0.5)
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"Mode{i + 1}_Decomposition.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_mode_reconstruction(
    original_data: np.ndarray, modes: np.ndarray, dates: np.ndarray, save_prefix: str = "VMD"
):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )

    reconstructed = np.sum(modes, axis=0)
    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(dates, original_data, label="Original True Value", color=(0.2, 0.2, 1.0), linewidth=1.0)
    plt.plot(
        dates,
        reconstructed,
        label="Reconstructed Value",
        color=(1.0, 0.7, 0.2),
        linestyle="--",
        linewidth=1.0,
    )
    plt.title("Original Data vs Modal Reconstruction", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "Modal_Reconstruction_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def static_weight_fusion(low_pred: np.ndarray, high_preds: Dict[int, np.ndarray]) -> np.ndarray:
    high_sum = np.sum(list(high_preds.values()), axis=0) if high_preds else np.array([])

    if len(low_pred) > 0 and len(high_sum) > 0:
        min_len = min(len(low_pred), len(high_sum))
        return low_pred[:min_len] + high_sum[:min_len]
    if len(low_pred) > 0:
        return low_pred
    return high_sum


def simple_residual_correction(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if len(y_true) < 3:
        return y_pred

    residual = y_true - y_pred
    max_correction = 0.0 * np.abs(y_pred)
    correction = np.clip(residual, -max_correction, max_correction)
    return y_pred + correction
