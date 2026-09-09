"""Fixed split and model seeds. Documented in registry later (M3 paso 3)."""

RANDOM_STATE = 42
TEST_SIZE = 0.15
N_FOLDS = 5
DECISION_THRESHOLD = 0.5
FN_COST = 5
FP_COST = 1

# Modest defaults for n≈1000. Optuna (paso 4) searches around these.
CATBOOST_PARAMS: dict[str, int | float | str | bool] = {
    "loss_function": "Logloss",
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "verbose": False,
}
