import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from scipy.signal import hilbert


def envelope_entropy(data: np.ndarray) -> float:
    analytic_signal = hilbert(np.asarray(data, dtype=float))
    amp = np.abs(analytic_signal)
    s = float(np.sum(amp))
    if s <= 0:
        return 0.0
    p = amp / s
    return float(-np.sum(p * np.log(p + 1e-12)))


def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(data, dtype=float)
    n = int(len(x))
    if n <= m + 2:
        return 0.0

    tol = float(r * np.std(x)) if np.std(x) > 0 else 1e-6

    def _phi(m_len: int) -> float:
        k = n - m_len
        if k <= 1:
            return 0.0
        X = np.array([x[i : i + m_len] for i in range(k)], dtype=float)
        diff = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        cnt = float(np.sum(diff <= tol) - k)
        denom = float(k * (k - 1))
        return cnt / denom if denom > 0 else 0.0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def _set_sci_style():
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


def plot_vmd_modes(modes: np.ndarray, dates: np.ndarray, save_prefix: str = "VMD"):
    _set_sci_style()

    modes = np.asarray(modes, dtype=float)
    n_modes = int(modes.shape[0])

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_modes):
        plt.figure(figsize=(6, 2), dpi=300)
        plt.plot(dates, modes[i], linewidth=1.0)
        plt.title(f"{save_prefix} Mode {i + 1}", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(linestyle="--", alpha=0.5)
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{save_prefix}_Mode_{i + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {os.path.abspath(save_path)}")
        plt.close()


def plot_mode_reconstruction(
    original_data: np.ndarray, modes: np.ndarray, dates: np.ndarray, save_prefix: str = "VMD"
):
    _set_sci_style()

    original_data = np.asarray(original_data, dtype=float)
    modes = np.asarray(modes, dtype=float)
    reconstructed = np.sum(modes, axis=0)

    save_dir = os.path.join("result", "Modal_Analysis", "Modal_Decomposition_Result")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(dates, original_data, label="Original", linewidth=1.0)
    plt.plot(dates, reconstructed, label="Reconstructed", linestyle="--", linewidth=1.0)
    plt.title(f"{save_prefix} Reconstruction", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{save_prefix}_Reconstruction.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(save_path)}")
    plt.close()


def static_weight_fusion(low_pred: np.ndarray, high_preds: Dict[int, np.ndarray]) -> np.ndarray:
    low_pred = np.asarray(low_pred, dtype=float).reshape(-1)
    if high_preds:
        high_sum = np.sum([np.asarray(v, dtype=float).reshape(-1) for v in high_preds.values()], axis=0)
    else:
        high_sum = np.asarray([], dtype=float)

    if low_pred.size and high_sum.size:
        n = min(low_pred.size, high_sum.size)
        return low_pred[:n] + high_sum[:n]
    if low_pred.size:
        return low_pred
    return high_sum


def simple_residual_correction(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < 3 or y_pred.size == 0:
        return y_pred

    n = min(y_true.size, y_pred.size)
    residual = y_true[:n] - y_pred[:n]

    max_correction = 0.0 * np.abs(y_pred[:n])
    correction = np.clip(residual, -max_correction, max_correction)
    out = y_pred.copy()
    out[:n] = y_pred[:n] + correction
    return out
