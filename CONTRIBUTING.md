# Contributing
1. Fork and branch; `pip install -r requirements-dev.txt`.
2. `python -m src.train && ruff check . && pytest -q` must pass (a test checks that `web/model.json` matches scikit-learn).
3. If you change the model, re-execute `notebooks/real_estate_valuation.ipynb` so the docs and outputs agree.
4. Open a pull request. Contributions are released under the MIT licence.
