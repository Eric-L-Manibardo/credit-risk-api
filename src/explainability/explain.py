"""CLI: global SHAP bar + two local waterfalls on the 85% pool (not the test)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import shap
from matplotlib import pyplot as plt

from src.explainability.explainer import (
    GLOBAL_SAMPLE,
    explain_row,
    make_explainer,
    mean_abs_shap,
    shap_values,
    to_explanation,
)
from src.features.engineering import as_catboost_frame
from src.features.pipeline import load_featured_loans
from src.model.registry import load_model, load_registry
from src.model.split import frame_to_xy, sealed_test_split

matplotlib.use("Agg")  # No display under WSL or CI: figures go straight to PNG.

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures" / "shap"


def save_global_bar(explanation: shap.Explanation, path: Path) -> Path:
    plt.figure()
    shap.plots.bar(explanation, max_display=12, show=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


def save_waterfall(explanation: shap.Explanation, index: int, path: Path) -> Path:
    shap.plots.waterfall(explanation[index], max_display=12, show=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


def save_beeswarm(explanation: shap.Explanation, path: Path) -> Path:
    """Same sample as the global bar: one point per row × feature, color = value."""
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


if __name__ == "__main__":
    registry = load_registry()
    model = load_model(registry)
    x, _y, ids = frame_to_xy(load_featured_loans())
    x_rest, _x_test, _y_rest, _y_test, ids_rest, _ids_test = sealed_test_split(x, _y, ids)
    x_cb = as_catboost_frame(x_rest)

    rng = np.random.default_rng(42)
    sample_n = min(GLOBAL_SAMPLE, len(x_cb))
    sample_idx = np.sort(rng.choice(len(x_cb), size=sample_n, replace=False))
    x_sample = x_cb.iloc[sample_idx]

    explainer = make_explainer(model)
    values, base = shap_values(explainer, x_sample)
    ranked = mean_abs_shap(values, list(x_cb.columns))
    print(f"Explaining {registry.get('version')} on {sample_n} rest-pool rows (not test)")
    print(f"base_value (log-odds) {base:.4f}")
    print("Top features by mean |SHAP|:")
    for item in ranked[:8]:
        print(f"  {item['feature']:24} {item['mean_abs_shap']:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    explanation = to_explanation(x_sample, values, base)
    print(f"  wrote {save_global_bar(explanation, FIGURES_DIR / 'shap_global_bar.png')}")
    print(f"  wrote {save_beeswarm(explanation, FIGURES_DIR / 'shap_beeswarm.png')}")

    proba = model.predict_proba(x_cb)[:, 1]
    i_hi = int(np.argmax(proba))
    i_lo = int(np.argmin(proba))
    x_local = x_cb.iloc[[i_hi, i_lo]]
    local_values, local_base = shap_values(explainer, x_local)
    local_exp = to_explanation(x_local, local_values, local_base)

    for pos, name, i in ((0, "high", i_hi), (1, "low", i_lo)):
        payload = explain_row(
            feature_names=list(x_cb.columns),
            feature_values=x_cb.iloc[i].to_numpy(),
            shap_row=local_values[pos],
            base_value=local_base,
            p_bad=float(proba[i]),
            loan_id=int(ids_rest.iloc[i]),
        )
        print(
            f"\n{name}-risk id={payload['id']} P(bad)={payload['p_bad']:.3f} "
            f"top={payload['contributions'][0]['feature']}"
        )
        out = FIGURES_DIR / f"shap_waterfall_{name}.png"
        print(f"  wrote {save_waterfall(local_exp, pos, out)}")
