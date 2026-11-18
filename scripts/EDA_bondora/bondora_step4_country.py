"""Bondora EDA step 4: does country still differ inside one origination year?

Reads data/raw/LoanData.csv. Does not write a database or a model.
The C engine fails on this file when selecting columns, so this uses Python.

The rate is the step-3 working mark: collection started within 12 months
of LoanDate. It is not the model target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "data" / "raw" / "LoanData.csv"
FIGURE_DIR = REPO / "docs" / "bondora" / "figures"
COLUMNS = ("ReportAsOfEOD", "LoanDate", "DefaultDate", "Country")
DAYS_PER_MONTH = 30.437
COUNTRY_ORDER = ("EE", "FI", "ES", "NL", "SK")
COMPARE = ("EE", "FI", "ES")
MIN_CELL = 100


def load_country_frame(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load country, origination year, and the 12-month collection mark."""
    frame = pd.read_csv(csv_path, usecols=list(COLUMNS), engine="python", on_bad_lines="skip")
    for column in ("ReportAsOfEOD", "LoanDate", "DefaultDate"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["year"] = frame["LoanDate"].dt.year
    frame["has_default_date"] = frame["DefaultDate"].notna()
    frame["months_observed"] = (frame["ReportAsOfEOD"] - frame["LoanDate"]).dt.total_seconds() / (
        86400 * DAYS_PER_MONTH
    )
    frame["months_to_default"] = (frame["DefaultDate"] - frame["LoanDate"]).dt.total_seconds() / (
        86400 * DAYS_PER_MONTH
    )
    watched = frame["months_observed"] >= 12
    within_12m = frame["months_to_default"].between(0, 12, inclusive="both")
    frame["default_within_12m"] = frame["has_default_date"] & watched & within_12m
    frame["eligible_12m"] = watched & frame["LoanDate"].notna()
    return frame


def _ordered(categories: pd.Index) -> list[str]:
    present = set(categories.astype(str))
    ordered = [name for name in COUNTRY_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per country. Mixes vintages on purpose."""
    grouped = frame.groupby("Country", dropna=False)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "ever_default_rate": grouped["has_default_date"].mean(),
        }
    )
    eligible = frame.loc[frame["eligible_12m"]]
    watched = eligible.groupby("Country", dropna=False)["default_within_12m"]
    table["n_watched_12m"] = watched.size()
    table["rate_12m_pooled"] = watched.mean()
    return table.reindex(_ordered(table.index))


def _counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(frame["year"], frame["Country"])
    return counts.reindex(columns=_ordered(counts.columns)).fillna(0).astype(int)


def _year_rates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """12-month mark on 2009–2022. Returns (n, rate) with country columns."""
    mature = frame.loc[frame["year"].between(2009, 2022) & frame["eligible_12m"]]
    grouped = mature.groupby(["year", "Country"])["default_within_12m"]
    counts = grouped.size().unstack()
    rates = grouped.mean().unstack()
    columns = [name for name in COUNTRY_ORDER if name in counts.columns]
    return counts.reindex(columns=columns), rates.reindex(columns=columns)


def report(frame: pd.DataFrame) -> str:
    """Pooled country rates, then the same 12-month rate inside each year."""
    counts, rates = _year_rates(frame)
    comparable = rates[list(COMPARE)].copy()
    small = counts[list(COMPARE)] < MIN_CELL
    comparable = comparable.mask(small)
    lines = [
        f"rows {len(frame)}",
        f"missing Country {int(frame['Country'].isna().sum())}",
        f"countries {sorted(frame['Country'].dropna().unique())}",
        "",
        "pooled by country (mixes vintages)",
        _pooled(frame).round(3).to_string(),
        "",
        "loan counts by year and country",
        _counts(frame).to_string(),
        "",
        "12-month default rate, 2009-2022",
        rates.round(3).to_string(),
        "",
        "n behind that rate",
        counts.fillna(0).astype(int).to_string(),
        "",
        f"EE FI ES rates, cells with n < {MIN_CELL} blanked",
        comparable.round(3).to_string(),
    ]
    return "\n".join(lines)


def write_figures(frame: pd.DataFrame, dest_dir: Path = FIGURE_DIR) -> list[Path]:
    """The pooled country rate, the count mix, and the rate inside one year."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pooled = _pooled(frame).dropna(subset=["n"])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(pooled))
    width = 0.36
    ax.bar(
        positions - width / 2,
        pooled["ever_default_rate"],
        width=width,
        label="DefaultDate ever set, all loans",
        color="#e45756",
    )
    ax.bar(
        positions + width / 2,
        pooled["rate_12m_pooled"],
        width=width,
        label="Within 12 months, loans watched ≥ 12 months",
        color="#4c78a8",
    )
    ax.set_xticks(positions, pooled.index.astype(str))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of that country's loans")
    ax.set_title("Country rates that pool every vintage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = dest_dir / "default_rate_by_country.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    counts = _counts(frame)
    years = counts.index.astype(int)
    colors = {
        "EE": "#4c78a8",
        "FI": "#f58518",
        "ES": "#e45756",
        "NL": "#54a24b",
        "SK": "#b279a2",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(counts))
    for country in counts.columns:
        heights = counts[country].to_numpy(dtype=float)
        ax.bar(years, heights, bottom=bottom, label=str(country), color=colors.get(str(country)))
        bottom = bottom + heights
    ax.set_ylabel("Loans")
    ax.set_title("Loans by origination year and country")
    ax.legend()
    fig.tight_layout()
    path = dest_dir / "loans_by_country_year.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    cell_n, cell_rate = _year_rates(frame)
    panel_n = cell_n[list(COMPARE)]
    panel = cell_rate[list(COMPARE)].mask(panel_n < MIN_CELL)
    grid = panel.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    image = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(COMPARE)), list(COMPARE))
    ax.set_yticks(range(len(panel.index)), [str(int(year)) for year in panel.index])
    ax.set_title("12-month mark inside one origination year")
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            label = "—" if np.isnan(value) else f"{value:.0%}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = dest_dir / "default_rate_country_year.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    frame = load_country_frame()
    print(report(frame))
    for path in write_figures(frame):
        print(f"figure {path}")


if __name__ == "__main__":
    main()
