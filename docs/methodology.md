# Methodology
* **Features:** the six raw inputs + `log_dist_mrt = log1p(dist_mrt)` + `dist_center_km` (haversine distance to Taipei
  Main Station, 25.0478 N 121.5170 E, used only as a fixed anchor).
* **Validation:** `RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)` — 25 folds; metrics on the original price scale.
* **Models:** mean baseline; linear regression and ridge (standardised); ridge on log target; random forest (500 trees,
  min leaf 2) with and without log target; HistGradientBoosting (log target); GradientBoostingRegressor (300 trees,
  depth 3, lr 0.05, min leaf 5, subsample 0.8) with squared and **Huber** loss on log target.
* **Log target:** prices are right-skewed; `TransformedTargetRegressor(func=log, inverse_func=exp)`.
* **Outliers:** robust z-score (median/MAD) > 3.5 flags one sale (117.5). It stays in training; a second evaluation
  skips it when scoring.
* **Quantile range:** separate GBR models with quantile loss at 0.05 and 0.95 on log price; coverage measured in the same CV.
* **Importance:** permutation importance on an 80/20 hold-out, 50 repeats, RMSE scoring.
* **Deployed model:** GBR Huber on log target (best CV RMSE), refit on all data by `src/train.py`.
