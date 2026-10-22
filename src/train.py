"""Train the deployed model and export it for the API (joblib) and the browser (JSON).

Deployed model: gradient-boosted trees (Huber loss, robust to outliers) on log(price),
plus two quantile models (5% / 95%) on log(price) giving a rough range (CV coverage 0.762, below the nominal 0.90).
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate

from src.features import FEATURES, ROOT, TARGET, load

SEED = 42
PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, subsample=0.8, random_state=SEED)


def make(loss="huber", alpha=0.9):
    return GradientBoostingRegressor(loss=loss, alpha=alpha, **PARAMS)


def export_gbr(m, X) -> dict:
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": t.threshold.tolist(),
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": t.value[:, 0, 0].tolist()})
    lr = m.learning_rate
    tree_sum = sum(e.predict(X[:1])[0] for e in m.estimators_[:, 0])
    base = float(m.predict(X[:1])[0] - lr * tree_sum)
    return {"base": base, "lr": lr, "trees": trees}


def cv_scores(X, y):
    def rmse(est, Xv, yv):
        return -float(np.sqrt(np.mean((np.exp(est.predict(Xv)) - np.exp(yv)) ** 2)))
    r = cross_validate(make(), X, np.log(y), cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED),
                       scoring={"rmse": rmse})
    return float(-r["test_rmse"].mean()), float(r["test_rmse"].std())


def train() -> dict:
    df = load()
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    ly = np.log(y)
    point, lo, hi = make().fit(X, ly), make("quantile", 0.05).fit(X, ly), make("quantile", 0.95).fit(X, ly)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"point": point, "lo": lo, "hi": hi}, ROOT / "models" / "model.joblib")
    export = {"features": FEATURES, "target": "log(price_per_ping)",
              "center": [25.0478, 121.5170],
              "point": export_gbr(point, X), "lo": export_gbr(lo, X), "hi": export_gbr(hi, X)}
    (ROOT / "web" / "model.json").write_text(json.dumps(export, separators=(",", ":")))
    m, s = cv_scores(X, y)
    metrics = {"cv_rmse_mean": m, "cv_rmse_std": s, "n": int(len(y))}
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=1))
