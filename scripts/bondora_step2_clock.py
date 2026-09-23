"""Bondora EDA step 2: how much of the book has had time to default.

Reads data/raw/LoanData.csv. Does not write a database or a model.
The C engine fails on this file when selecting columns, so this uses Python.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "LoanData.csv"
FIGURE_DIR = Path(__file__).resolve().parents[1] / "docs" / "bondora" / "figures"
COLUMNS = (
    "ReportAsOfEOD",
    "LoanDate",
    "Status",
    "ContractEndDate",
    "MaturityDate_Original",
)
DAYS_PER_MONTH = 30.437


def load_clock_frame(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the date and status columns used in EDA step 2."""
    frame = pd.read_csv(csv_path, usecols=list(COLUMNS), engine="python", on_bad_lines="skip")
    for column in ("ReportAsOfEOD", "LoanDate", "ContractEndDate", "MaturityDate_Original"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["year"] = frame["LoanDate"].dt.year
    frame["months_observed"] = (frame["ReportAsOfEOD"] - frame["LoanDate"]).dt.total_seconds() / (
        86400 * DAYS_PER_MONTH
    )
    frame["months_to_original_maturity"] = (
        frame["MaturityDate_Original"] - frame["LoanDate"]
    ).dt.total_seconds() / (86400 * DAYS_PER_MONTH)
    return frame


def _year_table(frame: pd.DataFrame) -> pd.DataFrame:
    status = pd.crosstab(frame["year"], frame["Status"])
    for name in ("Repaid", "Current", "Late"):
        if name not in status.columns:
            status[name] = 0
    status = status[["Repaid", "Current", "Late"]]
    status["n"] = status.sum(axis=1)
    current = frame["Status"].eq("Current")
    young = frame["months_observed"] < 12
    unfinished = frame["MaturityDate_Original"] > frame["ReportAsOfEOD"]

    def _share_current(status: pd.Series) -> float:
        return float(status.eq("Current").mean())

    extra = pd.DataFrame(
        {
            "pct_current": frame.groupby("year")["Status"].apply(_share_current),
            "pct_under_12m": young.groupby(frame["year"]).mean(),
            "pct_maturity_open": unfinished.groupby(frame["year"]).mean(),
            "current_months_p50": frame.loc[current].groupby("year")["months_observed"].median(),
        }
    )
    return status.join(extra)


def report(frame: pd.DataFrame) -> str:
    """Plain-text inventory of observation time."""
    lines = [
        f"rows {len(frame)}",
        f"snapshot {frame['ReportAsOfEOD'].dropna().iloc[0]}",
        f"missing LoanDate {int(frame['LoanDate'].isna().sum())}",
        "",
        _year_table(frame).round(3).to_string(),
        "",
        "current months observed",
        frame.loc[frame["Status"].eq("Current"), "months_observed"].describe().round(2).to_string(),
    ]
    return "\n".join(lines)


def write_figures(frame: pd.DataFrame, dest_dir: Path = FIGURE_DIR) -> list[Path]:
    """Volume, status mix, and how long Current loans have been watched."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest_dir.mkdir(parents=True, exist_ok=True)
    table = _year_table(frame)
    years = table.index.astype(int)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(years, table["n"], color="#4c78a8")
    ax.set_ylabel("Loans")
    ax.set_title("Loans by origination year")
    fig.tight_layout()
    path = dest_dir / "volume_by_year.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    repaid = table["Repaid"] / table["n"]
    late = table["Late"] / table["n"]
    current = table["Current"] / table["n"]
    ax.bar(years, repaid, label="Repaid")
    ax.bar(years, late, bottom=repaid, label="Late")
    ax.bar(years, current, bottom=repaid + late, label="Current")
    ax.set_ylabel("Share of the vintage")
    ax.set_title("Status on 23 May 2024, by origination year")
    ax.legend()
    fig.tight_layout()
    path = dest_dir / "status_by_year.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    current_rows = frame.loc[frame["Status"].eq("Current")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(current_rows["months_observed"].clip(upper=80), bins=40, color="#f58518")
    ax.set_xlabel("Months from LoanDate to the snapshot")
    ax.set_ylabel("Current loans")
    ax.set_title("How long Current loans have been observed")
    fig.tight_layout()
    path = dest_dir / "current_months_observed.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    frame = load_clock_frame()
    print(report(frame))
    for path in write_figures(frame):
        print(f"figure {path}")


if __name__ == "__main__":
    main()
