# Results
See the README table (repeated 5-fold CV). Additional numbers from the notebook:

| Check | Value |
|---|---|
| Deployed model CV RMSE (training script) | 7.136 ± 1.863 |
| CV RMSE when the extreme sale is not scored — Huber / squared / RF log / ridge log | 6.252 / 6.437 / 6.316 / 7.302 |
| 5–95% quantile range: empirical coverage | 0.762 ± 0.054 (nominal 0.90) |
| Quantile range: median width | 13.8 (× 10,000 NT$/ping) |
| 80/20 hold-out (deployed model) | RMSE 5.704, MAE 3.679, R² 0.806 |

Permutation importance (increase in hold-out RMSE): log_dist_mrt 2.768, dist_mrt 2.686, house_age 1.698, latitude
0.872, dist_center_km 0.530, longitude 0.403, n_stores 0.102, transaction_date 0.081.

**Interpretation.** Proximity to the MRT dominates; newer houses are worth more; location within the district matters.
The quantile models under-cover, so treat the range as indicative. With 414 sales, model differences among the top tree
ensembles are not meaningful.
