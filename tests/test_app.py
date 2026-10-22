import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_models
from src.features import FEATURES, TARGET, load
from src.predict import predict_json

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)
EX = {"transaction_date": 2013.25, "house_age": 13.3, "dist_mrt": 561.98, "n_stores": 5,
      "latitude": 24.98746, "longitude": 121.54391}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_shape_and_interval():
    r = client.post("/predict", json=EX)
    assert r.status_code == 200
    b = r.json()
    lo, hi = b["quantile_range_5_95"]
    assert 0 < lo < hi and 5 < b["price_per_ping_10k_twd"] < 120


def test_validation():
    assert client.post("/predict", json={**EX, "latitude": 30}).status_code == 422


def test_closer_to_mrt_is_not_cheaper_on_average():
    near = client.post("/predict", json={**EX, "dist_mrt": 100}).json()["price_per_ping_10k_twd"]
    far = client.post("/predict", json={**EX, "dist_mrt": 4000}).json()["price_per_ping_10k_twd"]
    assert near > far


def test_dataset():
    df = load()
    assert df.shape[0] == 414 and df[TARGET].min() > 0


def test_json_export_matches_sklearn():
    get_models()
    model = json.loads((ROOT / "web" / "model.json").read_text())
    df = load()
    m = get_models()
    X = df[FEATURES].to_numpy()
    ref = {k: np.exp(m[k].predict(X)) for k in ("point", "lo", "hi")}
    raw_cols = ["transaction_date", "house_age", "dist_mrt", "n_stores", "latitude", "longitude"]
    for i, row in enumerate(df[raw_cols].to_dict("records")):
        got = predict_json(model, row)
        for k in got:
            assert abs(got[k] - ref[k][i]) < 1e-6
