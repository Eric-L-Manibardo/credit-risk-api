# Roadmap

Built in vertical slices. Each item has a goal and a deliverable; later stages
depend on a working data layer.

| Stage | Focus | Status |
|-------|--------|--------|
| [Setup](#setup) | Project structure, dependencies, tooling | Done |
| [Data layer](#data-layer) | SQL + Pandas | Done |
| [Feature engineering](#feature-engineering) | Reproducible feature pipeline | Done |
| [Training](#model-training--evaluation) | CatBoost, metrics, model registry | In progress |
| [Explainability](#explainability-shap) | Global and per-prediction SHAP | Pending |
| [API](#fastapi-service) | REST scoring and explanations | Pending |
| [Release](#docker-cicd--readme) | Containers, CI, production docs | Pending |
| [Domain shift](#optional-cross-country--domain-shift) | Second country / schema | Optional |
| [Related service](#optional-related-service) | Monitoring, batch, or policy rules | Optional |

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

---

## Explainability (SHAP)

**Goal:** Global and per-prediction explanations.

**Deliverable:** SHAP module used by the API. Required for credit decisions.

---

## FastAPI service

**Goal:** Serve the model over REST.

**Deliverable:** Working API with integration tests.

- `POST /predict`
- `POST /predict/explain`
- `GET /health`
- `GET /model/info`

---

## Docker, CI/CD & README

**Goal:** Containerize the service and add GitHub Actions.

**Deliverable:** Multi-stage image, Compose, CI on pull requests.

---

## Optional: cross-country / domain shift

**Goal:** Run a second dataset (another country or schema) through the same
pipeline and document what breaks (mapping, calibration, SHAP, default
definition). A swapped CSV with a similar AUC is not sufficient.

**Deliverable:** Same ingest / features / serving code, second extract, short
shift report. Candidate sets: Taiwanese Credit, Home Credit, Lending Club
extract.

---

## Optional: related service

**Goal:** Add a service beside the API, not a second application.

| Add | Role |
|---|---|
| Monitoring / drift (PSI, calibration drop) | Detect model decay |
| Batch scoring (same library as the API) | Online vs batch |
| Policy rules (score + SHAP → review / reject) | Decisioning overlay |

**Deliverable:** One extra process in Compose, tested and documented.

---

## Out of scope

- Jupyter notebooks as production code
- Heavy MLOps (MLflow, Kubeflow)
- Frontend
- A second credit-scoring clone of this stack
