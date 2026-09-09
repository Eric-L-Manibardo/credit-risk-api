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
cross-validation, and `make evaluate` reports ranking, calibration, and
cost-per-client at every decision threshold. Results and the decisions they
drive are written up in
[docs/baseline_evaluation.md](docs/baseline_evaluation.md).

Persisted model artifacts, hyperparameter search, SHAP, the HTTP API, and
Docker/CI are next. See [ROADMAP.md](ROADMAP.md).

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
make train        # CatBoost baseline: CV and sealed-test metrics
make evaluate     # classification report + figures in reports/figures/
make test         # unit tests (in-memory SQLite)
make check        # lint + typecheck + tests
```

## Planned API

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| POST   | `/predict`          | Default probability + risk category      |
| POST   | `/predict/explain`  | Prediction + SHAP breakdown              |
| GET    | `/health`           | Health check                             |
| GET    | `/model/info`       | Model version, metrics, features used    |

## Project Structure

```
credit-risk-api/
├── data/                   # Local DB and raw extract (not committed)
│   └── raw/
├── src/
│   ├── data/               # Download, ingest, SQL access, validation
│   ├── features/           # Deterministic feature transform
│   ├── model/              # Split protocol, metrics, training, evaluation
│   ├── explainability/     # SHAP (upcoming)
│   └── api/                # FastAPI (upcoming)
├── tests/
│   ├── unit/
│   └── integration/
├── models/                 # Serialized artifacts (.cbm), gitignored
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

## Model Performance

To be published after training.

## License

MIT
