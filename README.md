<div align="center">

# 🏠 Taiwan Real Estate Valuation — House Price Prediction with Machine Learning (Python, scikit-learn, FastAPI) and a Live In-Browser Demo

[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-4f46e5?logo=github)](https://rishabh-bgp.github.io/real-estate-valuation-taiwan/)
[![CI](https://github.com/Rishabh-bgp/real-estate-valuation-taiwan/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishabh-bgp/real-estate-valuation-taiwan/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.161-009688?logo=fastapi&logoColor=white)
![Leaflet](https://img.shields.io/badge/map-Leaflet%20%2B%20OpenStreetMap-199900?logo=leaflet&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-0ea5e9)

**[▶ Open the live demo](https://rishabh-bgp.github.io/real-estate-valuation-taiwan/)**

<img src="docs/img/demo-light.png" alt="Screenshot of the Taiwan real estate valuation web app: OpenStreetMap map of Xindian with recorded sales coloured by price, property sliders, and an estimate of 37.5 × 10,000 NT$ per ping with a quantile range" width="900"/>

</div>

A complete **machine learning regression project** that predicts **house price per unit area (per ping)** in
**Xindian District, New Taipei City, Taiwan** from the public **UCI Real Estate Valuation dataset**: exploratory data
analysis, feature engineering (log distance to MRT, distance to Taipei Main Station), **repeated K-fold
cross-validation** of linear, ridge, random-forest and gradient-boosting models, log-target and outlier-robust (Huber)
training, permutation importance, quantile prediction ranges, a **FastAPI** REST API, **Docker**, **pytest**, **GitHub
Actions CI** and a modern **glassmorphism / bento** web app on **GitHub Pages** where the gradient-boosted trees run
entirely in your browser.

## 🎓 About this project
Developed as part of my training with **KF Property Network Pte Ltd (KFPN)** — a real estate agency associate business in
Singapore operating as a subsidiary of Knight Frank Singapore — which recommended this dataset, and submitted as my
**capstone for the NIELIT Ropar × IIT Ropar AI & ML programme** (the 6-Month Joint Certification Program in AI & ML by
[NIELIT Ropar](https://www.nielit.gov.in/ropar/) in collaboration with [IIT Ropar](https://www.iitrpr.ac.in/): live
sessions with NIELIT faculty, IIT Ropar guest lectures on NLP, deep learning and neural networks, hands-on labs,
project-based teaching and campus immersion at IIT Ropar).

> **Disclaimer:** this is a personal portfolio project built on a public dataset. It is not an official product of,
> and is not endorsed by, KFPN, Knight Frank or any other company, and it is not a property valuation service.

---

## 📈 Results

Repeated 5-fold cross-validation (5 repeats = 25 folds), all 414 sales, metrics on the original price scale
(10,000 NT$ per ping), **mean ± std across folds** — copied from the executed notebook:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **GBR, Huber loss, log target (deployed)** | **7.136 ± 1.863** | **4.551 ± 0.562** | **0.715 ± 0.094** |
| Random forest, log target | 7.159 ± 1.808 | 4.662 ± 0.575 | 0.714 ± 0.091 |
| Random forest | 7.240 ± 1.710 | 4.724 ± 0.561 | 0.706 ± 0.088 |
| GBR, squared loss, log target | 7.302 ± 1.754 | 4.710 ± 0.532 | 0.702 ± 0.090 |
| HistGradientBoosting, log target | 7.762 ± 1.737 | 5.056 ± 0.556 | 0.664 ± 0.090 |
| Ridge, log target | 8.021 ± 1.619 | 5.333 ± 0.567 | 0.640 ± 0.087 |
| Ridge | 8.021 ± 1.577 | 5.287 ± 0.569 | 0.640 ± 0.084 |
| Linear regression | 8.054 ± 1.588 | 5.307 ± 0.587 | 0.637 ± 0.084 |
| Mean baseline | 13.540 ± 1.613 | 10.758 ± 1.051 | −0.020 ± 0.028 |

* **Honest reading:** the top four tree models are within one standard deviation of each other; trees clearly beat the
  linear models.
* **Outliers:** one sale at 117.5 (row 270) is far above the rest. It is kept in training; not scoring it lowers the
  deployed model's CV RMSE to 6.252 ± 1.054 (Huber) vs 6.437 (squared loss).
* **Prediction range:** 5% / 95% quantile GBR models cover **76.2% ± 5.4%** of held-out sales — *below* the nominal 90%,
  so the app labels it as a rough range, not a calibrated interval. Median width 13.8.
* **Single 80/20 hold-out:** RMSE 5.704, MAE 3.679, R² 0.806 (one split, more optimistic than CV).
* **Permutation importance (hold-out, increase in RMSE):** log distance to MRT 2.768, distance to MRT 2.686, house age
  1.698, latitude 0.872, distance to Taipei Main Station 0.530, longitude 0.403, convenience stores 0.102, transaction
  date 0.081.
* **Browser ↔ Python parity:** on the live site, JavaScript predictions match scikit-learn for all 414 sales
  (max |Δ| ≈ 7 × 10⁻¹³ for the point and both quantile models).

Details: [docs/results.md](docs/results.md).

## 🖼️ Screenshots
| Light | Dark | Mobile |
|---|---|---|
| <img src="docs/img/demo-light.png" alt="Light theme" width="330"/> | <img src="docs/img/demo-dark.png" alt="Dark theme" width="330"/> | <img src="docs/img/demo-mobile.png" alt="Mobile layout" width="150"/> |

## 🏗️ Architecture
```mermaid
flowchart LR
  A[data/real_estate_valuation.csv<br/>UCI, CC BY 4.0] --> B[src/features.py<br/>log distance, distance to centre]
  B --> C[src/train.py<br/>GBR Huber + 5%/95% quantile GBR on log price]
  C --> D[models/model.joblib]
  C --> E[web/model.json<br/>300 trees × 3 models]
  D --> F[app/main.py<br/>FastAPI /predict]
  E --> G[web/app.js<br/>tree traversal in JS]
  G --> H[GitHub Pages + Leaflet/OSM map]
  F --> I[Docker / compose]
```

## 📂 Repository structure
```
notebooks/            self-contained EDA + modelling notebook (own README, requirements and data copy)
src/                  features.py, train.py (training + JSON export), predict.py (pure-Python JSON inference)
app/main.py           FastAPI: GET /health, POST /predict
tests/                pytest: API, validation, monotonic sanity check, JSON-vs-sklearn parity
web/                  static app: index.html, style.css, app.js, model.json, data.json
data/                 UCI dataset (original .xlsx and tidy .csv)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
gh-pages branch       published copy of web/
docs/                 extended documentation
```

## 🚀 Run locally
```bash
git clone https://github.com/Rishabh-bgp/real-estate-valuation-taiwan.git && cd real-estate-valuation-taiwan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train              # trains, writes models/ and web/model.json, prints CV RMSE
pytest -q
uvicorn app.main:app --reload    # http://localhost:8000/docs
python -m http.server -d web 8080
```
Notebook only: `cd notebooks && pip install -r requirements.txt && jupyter notebook` (see [notebooks/README.md](notebooks/README.md)).

## 🐳 Docker
```bash
docker compose up --build        # API → :8000 · web app → :8080
```

## 🔌 API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"transaction_date":2013.25,"house_age":13.3,"dist_mrt":561.98,"n_stores":5,"latitude":24.98746,"longitude":121.54391}'
```
Returns `price_per_ping_10k_twd`, `quantile_range_5_95`, `price_per_m2_twd` and the unit. See [docs/api.md](docs/api.md).

## 📚 Documentation
| Page | Contents |
|---|---|
| [docs/data.md](docs/data.md) | Dataset, units (ping, NT$), licence, citation |
| [docs/methodology.md](docs/methodology.md) | Features, models, CV, log target, outliers, quantiles |
| [docs/results.md](docs/results.md) | All results and interpretation |
| [docs/api.md](docs/api.md) | REST API |
| [docs/web-app.md](docs/web-app.md) | In-browser inference, UI, accessibility |
| [docs/deployment.md](docs/deployment.md) | Docker, CI, GitHub Pages |
| [docs/faq.md](docs/faq.md) | FAQ and limitations |

## 📝 Citation of the data
Yeh, I-C. & Hsu, T-K. (2018). Building real estate valuation models with comparative approach through case-based
reasoning. *Applied Soft Computing*, 65, 260–271. Dataset: UCI Machine Learning Repository, doi:10.24432/C5J30W (CC BY 4.0).

## ⚖️ Licence
Code and docs: MIT © 2026 Rishabh Aryan. Data: CC BY 4.0 (see above). Map tiles © OpenStreetMap contributors.

## 🔗 Other repositories by Rishabh Aryan

| Repository | Description | Live |
|---|---|---|
| [100-days-of-python](https://github.com/Rishabh-bgp/100-days-of-python) | My original, offline, no-API-key solutions to all 100 projects of the 100 Days of Code Python bootcamp curriculum… | — |
| [Agentic-Engineering-Ebook](https://github.com/Rishabh-bgp/Agentic-Engineering-Ebook) | Agentic Engineering is a practical notebook-book on specifying, generating, verifying, and refining multi-tenant… | — |
| [beed-ml-study](https://github.com/Rishabh-bgp/beed-ml-study) | Provenance audit and leakage-aware benchmark of seizure-type classifiers on the Bangalore EEG Epilepsy Dataset — IIT… | — |
| [Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction](https://github.com/Rishabh-bgp/Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction) | Brain–computer interfaces (BCIs) link neural activity to external devices, translating brain signals via algorithms… | — |
| [building-a-virtual-cell-in-python](https://github.com/Rishabh-bgp/building-a-virtual-cell-in-python) | A cell has a normal gene-activity pattern. Knock out one gene and that pattern changes. This project trains a small… | — |
| [Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML](https://github.com/Rishabh-bgp/Chronic-Kidney-Disease-Stage-Identification-in-HIV-Infected-Patients-Using-Supervised-ML) | ML is transforming Chronic Kidney Disease (CKD) detection in HIV patients, catching early asymptomatic stages to… | — |
| [cobratate](https://github.com/Rishabh-bgp/cobratate) | Cobratate, an esoteric imperative programming language. Tate-flavored syntax, ordinary semantics, and a Python… | [open](https://rishabh-bgp.github.io/cobratate/) |
| [coda-synth](https://github.com/Rishabh-bgp/coda-synth) | CoDA-Synth: stylometric detection of LLM-rewritten dark-web documents with sparse TF-IDF + LinearSVC. Research lab,… | [open](https://shale-ember-delta-kite.grok.me/) |
| [IIT-ROPAR-GENAI-FOUNDATIONS](https://github.com/Rishabh-bgp/IIT-ROPAR-GENAI-FOUNDATIONS) | Free open-source Generative AI foundations ebooks (2026) covering Claude, Grok, Gemini, Cursor, OpenAI and more —… | — |
| [iris-classification](https://github.com/Rishabh-bgp/iris-classification) | Iris flower classification in Python: scikit-learn model comparison, tuning and evaluation, FastAPI /predict API,… | [open](https://rishabh-bgp.github.io/iris-classification/) |
| [kbc-game](https://github.com/Rishabh-bgp/kbc-game) | Kaun Banega Crorepati (KBC) quiz game for the terminal in Python: 16-question prize ladder, safe levels and 4… | — |
| [knn-classifier](https://github.com/Rishabh-bgp/knn-classifier) | Simple K-Nearest Neighbors classifier on the classic Iris dataset using scikit-learn. Loads data, trains the model,… | — |
| [linear-regression-model-](https://github.com/Rishabh-bgp/linear-regression-model-) | Supervised linear regression on the scikit-learn diabetes dataset. A single feature (BMI) is used to predict… | — |
| [logistic-regression](https://github.com/Rishabh-bgp/logistic-regression) | This graduate capstone applies multi-class logistic regression to the Iris dataset. It includes EDA, stratified… | — |
| [logistics-website-capstone](https://github.com/Rishabh-bgp/logistics-website-capstone) | Logistics website capstone: Kargonaut Logistics, a premium freight site mixing claymorphism, glassmorphism and… | [open](https://rishabh-bgp.github.io/logistics-website-capstone/) |
| [lovelight-leap](https://github.com/Rishabh-bgp/lovelight-leap) | 💖 Lovelight Leap: a love-themed 2D platformer in vanilla JS canvas. 4 worlds, a boss you cheer up, original… | [open](https://rishabh-bgp.github.io/lovelight-leap/) |
| [machine-learning-part-1](https://github.com/Rishabh-bgp/machine-learning-part-1) | Machine learning in Python with scikit-learn: heart disease classification and medical insurance cost regression… | — |
| [Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail](https://github.com/Rishabh-bgp/Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail) | Agentic commerce lets shopping and seller agents place orders without showing operational truth. This IS case argues… | — |
| [morocco-city-aware-load-forecasting](https://github.com/Rishabh-bgp/morocco-city-aware-load-forecasting) | City-aware short-term load forecasting and peak detection on high-resolution Moroccan smart-meter data (5 zones).… | — |
| [NIELIT-DATA-SCIENCE](https://github.com/Rishabh-bgp/NIELIT-DATA-SCIENCE) | Hands-on NumPy, Pandas, and Matplotlib Jupyter notebooks for the NIELIT Data Science curriculum — practical,… | — |
| [NIELIT-MACHINE-LEARNING-PROJECTS](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS) | Nineteen end-to-end machine learning projects: classification, regression, clustering, and recommendation, with… | — |
| [NIELIT-PYTHON-](https://github.com/Rishabh-bgp/NIELIT-PYTHON-) | Open-source Python teaching notebooks: a language reference and five capstone programs, with detailed documentation.… | — |
| [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification) | Leakage, equivalence, conformal reliability and data efficiency on the UCI TCGA Pan-Cancer RNA-Seq dataset — IIT… | — |
| [Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks](https://github.com/Rishabh-bgp/Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks) | This notebook details the implementation and evaluation of a Bayesian perspective Early stage Event Prediction… | — |
| [prediction-as-power](https://github.com/Rishabh-bgp/prediction-as-power) | Laboratory and working paper on prediction as power in European electricity markets. A DE-like hourly simulator… | — |
| [snakes-and-ladders](https://github.com/Rishabh-bgp/snakes-and-ladders) | 🐍🪜 Original Snakes & Ladders browser game: hand-drawn SVG snakes and ladders, 3D dice, 2-4 humans or bots, sound,… | [open](https://rishabh-bgp.github.io/snakes-and-ladders/) |
| [spam-scan](https://github.com/Rishabh-bgp/spam-scan) | Explainable spam, phishing & Indian-scam detector for SMS/email (English, Hindi, Hinglish): TF-IDF + Logistic… | [open](https://spam-scan.onrender.com) |
| [vedic-math-hybrid-kt](https://github.com/Rishabh-bgp/vedic-math-hybrid-kt) | Hybrid Vedic Math AI: rule-based sutras (Urdhva-Tiryagbhyam, Nikhilam, Yavadunam) plus RF classifier that cleans… | — |
| [wind-turbine-power-model](https://github.com/Rishabh-bgp/wind-turbine-power-model) | Single-turbine power curves + wind-farm simulation. Weibull wind stats, hourly NASA-style speed data, and… | — |
| [zoo-website-capstone](https://github.com/Rishabh-bgp/zoo-website-capstone) | Capstone project: a fictional zoo website built twice — a beginner-style simple version and a cinematic advanced… | [open](https://rishabh-bgp.github.io/zoo-website-capstone/) |
