# Credit Risk API

Credit default scoring service: CatBoost classification, SHAP explanations, FastAPI.

## Architecture

```mermaid
graph LR
    A[Raw Data] --> B[SQLite DB]
    B --> C[SQL Queries]
    C --> D[Feature Engineering]
    D --> E[CatBoost Model]
    E --> F[SHAP Explainer]
    F --> G[FastAPI]
    G --> H[POST /predict]
    G --> I[POST /predict/explain]
    G --> J[GET /health]
    G --> K[GET /model/info]
```

## Status

The data layer and feature pipeline are in place: German Credit is ingested
into SQLite, accessed only through SQLAlchemy, validated, and transformed
with a deterministic `create_features` used for training and scoring.

A CatBoost baseline trains against a sealed test set with 5-fold stratified
cross-validation. `make train` writes `models/baseline-v1.cbm` (gitignored)
and `models/registry.json` (committed: metrics, features, operating point).
`make tune` searches hyperparameters on CV PR-AUC with the test set closed,
then persists the winner as `models/tuned-v1.cbm`. `make evaluate` reports
ranking, calibration, and cost-per-client at every decision threshold.

How the baseline behaves:
[docs/baseline_evaluation.md](docs/baseline_evaluation.md).
What the Optuna search changed:
[docs/tuned-v1_evaluation.md](docs/tuned-v1_evaluation.md).
How the model attributes a score:
[docs/shap_explainability.md](docs/shap_explainability.md).

The HTTP API serves health, model info, scoring, and per-application
SHAP. Docker/CI are next. See [ROADMAP.md](ROADMAP.md).

## Tech Stack

| Component        | Technology                     |
|-----------------|--------------------------------|
| Data Storage    | SQLite (dev) / PostgreSQL (prod) |
| Data Processing | Pandas, SQLAlchemy             |
| ML Model        | CatBoost                       |
| Explainability  | SHAP                           |
| API Framework   | FastAPI                        |
| Validation      | Pydantic                       |
| Testing         | pytest                         |
| Containerization| Docker, Docker Compose         |
| CI/CD           | GitHub Actions                 |
| Package manager | uv                             |
| Linting         | Ruff, mypy                     |

## Quick Start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Eric-L-Manibardo/credit-risk-api.git
cd credit-risk-api

make dev          # .venv + dependencies
make ingest       # download extract → SQLite
make validate     # domain checks on the loans table
make train        # CatBoost baseline → models/baseline-v1.cbm + registry.json
make tune         # Optuna on CV PR-AUC (sealed test closed) → models/tuned-v1.cbm
make evaluate     # classification report + figures in reports/figures/evaluation/
make explain      # SHAP figures in reports/figures/shap/ (rest pool, not test)
make run          # FastAPI on :8000 (needs the .cbm named in registry.json)
make test         # unit tests (in-memory SQLite)
make check        # lint + typecheck + tests
```

## API

`make run` serves all four endpoints. Open `/docs` for the interactive
schema.

A loan is never sent raw into the model. One request follows this path:

```
JSON body → Pydantic → create_features → as_catboost_frame
        → predict_proba  (+ TreeExplainer on /explain)  → JSON
```

| Step | What happens |
|------|----------------|
| JSON body | The client sends the application (income-style fields, amount, duration, …). There is **no** default label (`credit_class`): that is what we are estimating. |
| Pydantic | Checks types, allowed categories, and numeric ranges. An unknown checking-account status or an age of 12 stops here with **422**. Nothing is scored. |
| `create_features` | Adds the same derived columns used at training time (`credit_per_month`, `log_credit_amount`, `age_bin`). Train and serve must see the same columns. |
| `as_catboost_frame` | Casts categoricals to text. CatBoost (and SHAP) were trained on native categories, not one-hot encodings. |
| `predict_proba` | The model already loaded at startup returns **P(bad)**: the probability this applicant is a poor credit risk. |
| TreeExplainer | Only on `/predict/explain`. It does **not** return a plot. It returns numbers: how much each feature pushed the score, in log-odds. The waterfall picture is just those numbers drawn; the API returns the list. |
| JSON | `P(bad)` plus a decision at the registry threshold (`approve` / `review`). Explain adds `base_value`, `units: log_odds`, and `contributions`. |

`GET /health` only answers “is the process up and did the model load?”.
`GET /model/info` answers “which artifact, which threshold, which metrics?”.

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| POST   | `/predict`          | Default probability + risk category      |
| POST   | `/predict/explain`  | Prediction + SHAP breakdown              |
| GET    | `/health`           | Health check                             |
| GET    | `/model/info`       | Model version, metrics, features used    |

## Dataset

Training data is the OpenML German Credit extract (`credit-g`, 1994).
Columns such as `personal_status` (gendered marital status) and
`foreign_worker` are **features of that dataset, not a policy I would
deploy**. A production model in the EU would drop or tightly constrain
protected attributes. They remain in the schema so train, SHAP, and
`/predict` see the same published columns.

## Project Structure

```
credit-risk-api/
├── data/                   # Local DB and raw extract (not committed)
│   └── raw/
├── src/
│   ├── data/               # Download, ingest, SQL access, validation
│   ├── features/           # Deterministic feature transform
│   ├── model/              # Split protocol, metrics, training, evaluation
│   ├── explainability/     # SHAP (log-odds TreeExplainer)
│   └── api/                # FastAPI (health, info, predict, explain)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/                   # Evaluation notes and curated figures
├── models/                 # {version}.cbm (gitignored) + registry.json
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

## Model Performance

Current artifact is `tuned-v1` (20 Optuna trials, objective = mean 5-fold
PR-AUC). Operating point is still 0.5; the Hofmann-cost optimum (0.25) is
a product decision, not applied yet.

| Metric | Baseline CV | Tuned CV | Tuned test |
|---|---|---|---|
| ROC-AUC | 0.788 ± 0.024 | 0.792 ± 0.021 | 0.774 |
| PR-AUC | 0.609 ± 0.047 | **0.633 ± 0.065** | 0.674 |
| Brier | 0.179 ± 0.004 | 0.184 ± 0.010 | 0.191 |
| Recall (`bad`) @ 0.5 | 0.694 ± 0.078 | 0.741 ± 0.102 | 0.689 |
| Cost / client @ 0.5 | 0.635 ± 0.088 | 0.595 ± 0.111 | 0.673 |

The search improved the metric it was allowed to see (CV PR-AUC). Test
PR-AUC moved 0.692 → 0.674; with ~45 defaults that is inside the noise,
so it does not veto the winner. Full manifest:
[models/registry.json](models/registry.json).

- Baseline figures: [docs/baseline_evaluation.md](docs/baseline_evaluation.md)
- Tuned-v1 figures and Optuna search: [docs/tuned-v1_evaluation.md](docs/tuned-v1_evaluation.md)
- SHAP (log-odds, rest pool): [docs/shap_explainability.md](docs/shap_explainability.md)

## License

MIT
