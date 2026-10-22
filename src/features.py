"""Data loading and feature engineering for the UCI Real Estate Valuation dataset."""
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "real_estate_valuation.csv"
RAW = ["transaction_date", "house_age", "dist_mrt", "n_stores", "latitude", "longitude"]
TARGET = "price_per_ping"
# Reference point: Taipei Main Station (used only as a fixed geographic anchor).
CENTER = (25.0478, 121.5170)
FEATURES = RAW + ["log_dist_mrt", "dist_center_km"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_dist_mrt"] = np.log1p(out["dist_mrt"])
    out["dist_center_km"] = [haversine_km(a, b, *CENTER) for a, b in zip(out["latitude"], out["longitude"])]
    return out


def load() -> pd.DataFrame:
    return add_features(pd.read_csv(DATA_FILE))
