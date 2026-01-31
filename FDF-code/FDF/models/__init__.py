# models/__init__.py
from .arima import arima
from .xgboost import xgboost
from .gru import gru
from .dlinear import dlinear
from .nbeats import nbeats
from .tcn import tcn
from .patchtst import patchtst

LOW_MODEL_REGISTRY = {
    "ARIMA": arima,
    "XGBoost": xgboost,
    "DLinear": dlinear,
    "NBEATS": nbeats,
    "GRU": gru,
    "TCN": tcn,
    "PatchTST": patchtst,
}

HIGH_MODEL_REGISTRY = {
    "ARIMA": arima,
    "XGBoost": xgboost,
    "DLinear": dlinear,
    "NBEATS": nbeats,
    "GRU": gru,
    "TCN": tcn,
    "PatchTST": patchtst,
}
