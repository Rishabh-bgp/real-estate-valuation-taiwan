"""FastAPI service for the Taiwan real estate valuation model."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import features

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
PING_M2 = 3.305785
app = FastAPI(title="Real Estate Valuation API (Xindian, New Taipei)", version="1.0.0")
_models = None


def get_models():
    global _models
    if _models is None:
        if not MODEL.exists():
            from src.train import train
            train()
        _models = joblib.load(MODEL)
    return _models


class Property(BaseModel):
    transaction_date: float = Field(..., ge=2012.0, le=2014.0, examples=[2013.25],
                                    description="Decimal year, e.g. 2013.25 = March 2013")
    house_age: float = Field(..., ge=0, le=60, examples=[13.3])
    dist_mrt: float = Field(..., ge=0, le=10000, examples=[561.98], description="Metres to nearest MRT station")
    n_stores: int = Field(..., ge=0, le=15, examples=[5])
    latitude: float = Field(..., ge=24.9, le=25.1, examples=[24.98746])
    longitude: float = Field(..., ge=121.4, le=121.6, examples=[121.54391])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Property):
    x = np.array([features(p.model_dump())])
    m = get_models()
    point, lo, hi = (float(np.exp(m[k].predict(x)[0])) for k in ("point", "lo", "hi"))
    return {"price_per_ping_10k_twd": round(point, 3),
            "quantile_range_5_95": [round(lo, 3), round(hi, 3)],
            "price_per_m2_twd": round(point * 10000 / PING_M2, 1),
            "unit": "10,000 New Taiwan Dollars per ping (1 ping = 3.3058 m2)"}
