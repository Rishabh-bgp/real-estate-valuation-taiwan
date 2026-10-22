# API (FastAPI)
`uvicorn app.main:app` → docs at `/docs`.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| POST | `/predict` | body: `transaction_date` (2012–2014), `house_age` (0–60), `dist_mrt` (0–10000 m), `n_stores` (0–15), `latitude` (24.9–25.1), `longitude` (121.4–121.6) |

Response: `price_per_ping_10k_twd`, `quantile_range_5_95` [lo, hi], `price_per_m2_twd`, `unit`. Out-of-range input → 422.
