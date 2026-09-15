"""Fixed split and model seeds. Persisted in the model registry (M3 paso 3)."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.15
N_FOLDS = 5
DECISION_THRESHOLD = 0.5
FN_COST = 5
FP_COST = 1

OPTUNA_N_TRIALS = 20

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "baseline.cbm"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Modest defaults for n≈1000. Optuna (make tune) searches around these.
CATBOOST_PARAMS: dict[str, int | float | str | bool] = {
    "loss_function": "Logloss",
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "verbose": False,
}
