from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


DATA_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
TARGET = "Machine failure"
NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
FEATURES = ["Type", *NUMERIC]
FORBIDDEN = ["UDI", "Product ID", TARGET, "TWF", "HDF", "PWF", "OSF", "RNF"]
ENGINEERED = ["Temperature gap [K]", "Mechanical power [W]", "Wear torque [min Nm]"]

def validate_features(frame):
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Input contains no observations")
    out = frame[FEATURES].copy()
    if not out["Type"].isin(["H", "L", "M"]).all():
        raise ValueError("Type must be H, L, or M")
    for column in NUMERIC:
        out[column] = pd.to_numeric(out[column], errors="raise")
    values = out[NUMERIC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sensor values must be finite and non-missing")
    if (values[:, :3] <= 0).any() or (values[:, 3:] < 0).any():
        raise ValueError("Temperatures/speed must be positive; torque/wear nonnegative")
    return out

class SensorFeatures:
    """An explicit allowlist prevents target/identifier columns entering a model."""
    def __init__(self, engineered=False):
        self.engineered = engineered

    def fit(self, X, y=None):
        validate_features(X)
        self.n_features_in_ = len(FEATURES)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        out = validate_features(X)
        if self.engineered:
            out[ENGINEERED[0]] = out[NUMERIC[1]] - out[NUMERIC[0]]
            out[ENGINEERED[1]] = out[NUMERIC[2]] * out[NUMERIC[3]] * 2 * np.pi / 60
            out[ENGINEERED[2]] = out[NUMERIC[3]] * out[NUMERIC[4]]
        return out

def load_benchmark(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch; do not silently replace the benchmark")
    frame = pd.read_csv(path)
    validate_features(frame)
    if len(frame) != 10000 or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Unexpected benchmark shape or labels")
    if not frame["UDI"].is_unique or not np.array_equal(frame["UDI"], np.arange(1, 10001)):
        raise ValueError("Expected original UDI order 1..10000")
    return frame

def split_indices(n=10000):
    if n != 10000:
        raise ValueError("This benchmark protocol requires the original 10000 rows")
    # UDI is an ordering key, not a timestamp or machine identifier.
    return {
        "train": np.arange(0, 5500),
        "selection": np.arange(5500, 7000),
        "policy": np.arange(7000, 8000),
        "test": np.arange(8000, 10000),
    }
