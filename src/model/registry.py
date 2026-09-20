"""Persist and load the trained model and its metadata.

The ``.cbm`` binary is gitignored (regenerate with ``make train`` or
``make tune``). The filename is the registry version
(``baseline-v1.cbm``, ``tuned-v1.cbm``). ``registry.json`` is committed:
it is the manifest the API and ``/model/info`` read at startup. A JSON
without the binary cannot score; a binary without the JSON does not
document the operating point.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

from src.features.engineering import CATEGORICAL_FEATURES
from src.model.config import (
    CATBOOST_PARAMS,
    DECISION_THRESHOLD,
    FN_COST,
    FP_COST,
    MODELS_DIR,
    REGISTRY_PATH,
    artifact_path,
)

REGISTRY_KEYS = frozenset(
    {
        "version",
        "trained_at",
        "model_path",
        "features",
        "cat_features",
        "n_train",
        "n_test",
        "cv_metrics",
        "test_metrics",
        "threshold",
        "cost_ratio",
        "catboost_params",
    }
)


def _params_for_registry(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("params", CATBOOST_PARAMS)
    return {key: value for key, value in raw.items() if key != "verbose"}


def save_model(
    model: CatBoostClassifier,
    *,
    result: dict[str, Any],
    model_path: Path | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> Path:
    """Write the .cbm and registry.json. Returns the registry path.

    Default filename is ``{version}.cbm`` so a baseline fit cannot
    overwrite the tuned artifact (or the other way around).
    """
    version = str(result.get("version", "baseline-v1"))
    path = model_path if model_path is not None else artifact_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    registry: dict[str, Any] = {
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_path": path.name,
        "features": result["feature_names"],
        "cat_features": list(CATEGORICAL_FEATURES),
        "n_train": result["n_rest"],
        "n_test": result["n_test"],
        "cv_metrics": result["cv"],
        "test_metrics": {
            key: value
            for key, value in result["test"].items()
            if key not in {"tn", "fp", "fn", "tp"}
        },
        "threshold": DECISION_THRESHOLD,
        "cost_ratio": {"fn": FN_COST, "fp": FP_COST},
        "catboost_params": _params_for_registry(result),
    }
    if "tuning" in result:
        registry["tuning"] = result["tuning"]
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry_path


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Read the registry manifest. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Registry {path} not found. Run: make train or make tune")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Registry {path} is not a JSON object")
    return data


def load_model(
    registry: dict[str, Any] | None = None,
    *,
    models_dir: Path | None = None,
) -> CatBoostClassifier:
    """Load the .cbm named in the registry.

    ``model_path`` in the JSON is a filename, portable across machines.
    Resolve it against ``models_dir`` (default: ``models/``), so tests can
    round-trip in a temp directory without touching the real artifact.
    """
    if registry is None:
        registry = load_registry()
    root = models_dir if models_dir is not None else MODELS_DIR
    model_path = root / registry["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Run: make train or make tune")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model
