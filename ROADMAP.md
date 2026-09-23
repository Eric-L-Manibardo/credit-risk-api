# Roadmap

Two books live in this repo. The German Credit API is finished and stays
the served model. The current work is the Bondora loan book: euros,
2009–2024, read before any schema or second model.

| Book | What it is | Status |
|------|------------|--------|
| [German Credit](#german-credit--finished) | PoC scoring API (OpenML `credit-g`, 1994) | Finished |
| [Bondora](#bondora--in-progress) | Consumer loans in euros. EDA before a schema | In progress |

Public notes: [docs/README.md](docs/README.md).

---

## German Credit — finished

The pipeline below is the proof of concept. It is not the template for
Bondora. Data, features, CatBoost, SHAP, FastAPI, Docker, and CI are closed.

| Stage | Focus | Status |
|-------|--------|--------|
| [Setup](#setup) | Project structure, dependencies, tooling | Done |
| [Data layer](#data-layer) | SQL + Pandas | Done |
| [Feature engineering](#feature-engineering) | Reproducible feature pipeline | Done |
| [Training](#model-training--evaluation) | CatBoost, metrics, model registry | Done |
| [Explainability](#explainability-shap) | Global and per-prediction SHAP | Done |
| [API](#fastapi-service) | REST scoring and explanations | Done |
| [Release](#docker-cicd--readme) | Containers, CI, production docs | Done |

```
[CSV] → [SQLite DB] → [SQL Queries] → [Pandas DF] → [Feature Engineering] → [CatBoost Model]
                                                                                    ↓
                                                              [FastAPI] ← [SHAP Explainer]
                                                                 ↓
                                                         [Docker + CI/CD]
```

---

## Setup

**Goal:** Repository layout and development tooling.

**Deliverable:** Installable project with uv (`make install` / `make dev`).

---

## Data layer

**Goal:** Load the loan extract into SQLite and access it only through SQL.

**Deliverable:** Populated SQLite database, SQLAlchemy load/query functions,
validation checks, unit tests.

Dataset: German Credit (OpenML `credit-g`).

---

## Feature engineering

**Goal:** Derived features (ratios, bins, and related signals) as code.

**Deliverable:** Testable, reproducible feature module. Categorical columns
stay native for CatBoost.

---

## Model training & evaluation

**Goal:** CatBoost classifier with train/val/test split, hyperparameter
tuning, and persisted artifacts.

**Deliverable:** Trained model, recorded metrics (AUC-ROC, precision-recall,
calibration), `.cbm` serialization, JSON model registry.

Baseline artifacts are in `models/` (`make train` → `baseline-v1.cbm`).
Hyperparameter search is `make tune`: Optuna on CV PR-AUC, sealed test
closed, winner persisted as `tuned-v1.cbm`.

---

## Explainability (SHAP)

**Goal:** Global and per-prediction explanations.

**Deliverable:** SHAP module used by the API. Required for credit decisions.

CatBoost categorical splits force TreeExplainer in `tree_path_dependent`
mode, so SHAP is in **log-odds**, not probability. `make explain` ranks features on a 200-row sample of the 85 % rest pool
(sealed test stays closed) and writes a global bar plus two local
waterfalls. Notes: [docs/german_credit/shap_explainability.md](docs/german_credit/shap_explainability.md).
The per-row dict is the payload `POST /predict/explain` returns.

---

## FastAPI service

**Goal:** Serve the model over REST.

**Deliverable:** Working API with integration tests.

The registered artifact loads once at startup. `POST /predict` scores
one application at the registry threshold. `POST /predict/explain`
returns the same score plus a local SHAP breakdown in log-odds.

- `POST /predict`
- `POST /predict/explain`
- `GET /health`
- `GET /model/info`

---

## Docker, CI/CD & README

**Goal:** Containerize the service and add GitHub Actions.

**Deliverable:** Multi-stage image, Compose, CI on pull requests.

`Dockerfile` builds with uv; the runtime image is slim Python plus
libgomp (CatBoost). Compose mounts `./models` because the `.cbm` is
gitignored. PRs run Ruff, mypy, pytest, and `docker build`
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Bondora — in progress

**Goal:** Decide the population, the label, and the origination columns
in writing before any table or model. The German Credit pipeline is not
copied onto this file.

**Deliverable so far:** Public EDA notes under `docs/bondora/`. The
working mark is collection started within 12 months of `LoanDate`, on
vintages 2009–2022. It is not the model target.

| Step | Note | Status |
|------|------|--------|
| 1. What a row is | [docs/bondora/eda1.md](docs/bondora/eda1.md) | Done |
| 2. The clock stops | [docs/bondora/eda2.md](docs/bondora/eda2.md) | Done |
| 3. `DefaultDate` is not `Status` | [docs/bondora/eda3.md](docs/bondora/eda3.md) | Done |
| 4. Country inside the same year | — | Next |
| 5. Column inventory | — | Open |
| 6. Associations that keep their sign | — | Open |
| 7. Columns that are the outcome | — | Open |
| 8. Closing note: population, label, columns | — | Open |

Download, local only (the CSV is gitignored):

```bash
uv run python -m src.data.download bondora
```

Needs a Kaggle token in `~/.kaggle/access_token`. It does not touch the
German Credit database or the API.

`make ingest`, `make train`, and `make run` still mean German Credit.
Bondora has no schema, no model, and no second service yet. Those wait
on the closing note.

---

## Later

A service beside the German Credit API (drift monitoring, batch scoring,
or a score-plus-SHAP policy) stays unscheduled. It is not the current
work.

---

## Out of scope

- Jupyter notebooks as production code
- Heavy MLOps (MLflow, Kubeflow)
- Frontend
- A second credit-scoring clone of the German Credit stack
