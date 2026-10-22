"""Pure-Python inference from the exported JSON (same algorithm as web/app.js)."""
import math

import numpy as np

from src.features import CENTER, haversine_km


def _tree(t, x):
    # scikit-learn compares float32-cast features against float64 thresholds.
    x = [float(np.float32(v)) for v in x]
    n = 0
    while t["l"][n] != -1:
        n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
    return t["v"][n]


def _gbr(m, x):
    return m["base"] + m["lr"] * sum(_tree(t, x) for t in m["trees"])


def features(raw: dict) -> list:
    d = raw["dist_mrt"]
    return [raw["transaction_date"], raw["house_age"], d, raw["n_stores"], raw["latitude"], raw["longitude"],
            math.log1p(d), haversine_km(raw["latitude"], raw["longitude"], *CENTER)]


def predict_json(model: dict, raw: dict) -> dict:
    x = features(raw)
    return {k: math.exp(_gbr(model[k], x)) for k in ("point", "lo", "hi")}
