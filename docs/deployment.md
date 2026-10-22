# Deployment
* **Docker:** `docker build -t rev-api . && docker run -p 8000:8000 rev-api` (model trained during build).
  The Docker files were written but not built in the authoring environment (no Docker available there).
* **docker-compose:** API on :8000, static web app via nginx on :8080.
* **CI:** `.github/workflows/ci.yml` — Python 3.12/3.13, `ruff check .`, `python -m src.train`, `pytest`.
* **GitHub Pages:** served from the `gh-pages` branch (a copy of `web/`). To update, copy `web/*` to that branch and push.
