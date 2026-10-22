# Notebooks — Taiwan real estate valuation

`real_estate_valuation.ipynb` is **self-contained**: it does not import anything from `src/` or `app/`, and reads its
own copy of the data from `notebooks/data/real_estate_valuation.csv`.

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook real_estate_valuation.ipynb
```
Contents: EDA (distributions, map, correlations, outliers) → repeated 5-fold CV of 9 models → outlier-robustness check
→ quantile-interval coverage → hold-out diagnostics → permutation importance → conclusions. Full run ≈ 1–2 minutes.
