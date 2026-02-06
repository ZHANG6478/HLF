# models/gru.py
import numpy as np
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from common import create_sequence_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        l2_lambda: float = 1e-4,
    ):
        super().__init__()
        self.l2_lambda = float(l2_lambda)

        self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.fc1 = nn.Linear(int(hidden_size), int(hidden_size))
        self.fc2 = nn.Linear(int(hidden_size), 1)
        self.dropout = nn.Dropout(float(dropout))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

    def l2_regularization(self) -> torch.Tensor:
        reg = 0.0
        for p in self.parameters():
            reg = reg + torch.norm(p, 2)
        return self.l2_lambda * reg


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-5, restore_best_weights: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False


def _fit_gru_on_history(
    history_raw: np.ndarray,
    scaler: MinMaxScaler,
    seq_len: int,
    epochs: int,
    batch_size: int,
    verbose: bool,
) -> Optional[GRUModel]:
    history_raw = np.asarray(history_raw, dtype=float)
    hist_scaled = scaler.transform(history_raw.reshape(-1, 1)).flatten()

    X_all, y_all = create_sequence_data(hist_scaled, int(seq_len))
    if len(X_all) == 0:
        return None

    split = int(len(X_all) * 0.8)
    X_train = X_all[:split]
    y_train = y_all[:split]
    X_val = X_all[split:]
    y_val = y_all[split:]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(2),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=False)

    model = GRUModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early = EarlyStopping(patience=8, min_delta=1e-5, restore_best_weights=True)

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb) + model.l2_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(X_val) == 0:
                val_loss = 0.0
            else:
                xv = torch.FloatTensor(X_val).unsqueeze(2).to(DEVICE)
                yv = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
                val_loss = float(criterion(model(xv), yv).item())

        scheduler.step(val_loss)

        if verbose:
            print(f"val_loss={val_loss:.6f}")

        if early.step(val_loss, model):
            break

    return model


def gru(
    train_series,
    test_series,
    seq_len: int = 14,
    epochs: int = 100,
    batch_size: int = 32,
    refit_interval: Optional[int] = 7,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    train_series = np.asarray(train_series, dtype=float)
    test_series = np.asarray(test_series, dtype=float)

    n_test = len(test_series)
    if n_test == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    seq_len = int(seq_len)
    if len(train_series) <= seq_len:
        raise ValueError(f"train_series length ({len(train_series)}) must be > seq_len ({seq_len})")

    do_refit = (refit_interval is not None) and (int(refit_interval) > 0)
    refit_steps = int(refit_interval) if do_refit else n_test

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series.reshape(-1, 1))

    history = train_series.tolist()
    preds = np.zeros(n_test, dtype=float)

    model = _fit_gru_on_history(
        np.asarray(history, dtype=float),
        scaler=scaler,
        seq_len=seq_len,
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=bool(verbose),
    )

    for i in range(n_test):
        if do_refit and i > 0 and (i % refit_steps == 0):
            model = _fit_gru_on_history(
                np.asarray(history, dtype=float),
                scaler=scaler,
                seq_len=seq_len,
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=bool(verbose),
            )

        win = np.asarray(history[-seq_len:], dtype=float).reshape(-1, 1)
        win_scaled = scaler.transform(win).flatten().reshape(1, seq_len, 1)

        if model is None:
            y_hat = float(history[-1])
        else:
            model.eval()
            with torch.no_grad():
                xt = torch.FloatTensor(win_scaled).to(DEVICE)
                y_hat_scaled = float(model(xt).detach().cpu().numpy().reshape(-1)[0])
            y_hat = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])

        preds[i] = y_hat
        history.append(float(test_series[i]))

    if preds.shape[0] != len(test_series):
        raise RuntimeError(f"GRU len check failed: preds={preds.shape[0]} test={len(test_series)}")

    return preds, test_series


gru_predict = gru
