"""Bondora EDA step 3: what DefaultDate is, next to Status.

Reads data/raw/LoanData.csv. Does not write a database or a model.
The C engine fails on this file when selecting columns, so this uses Python.

The printed 12-month mark is a working label for later EDA charts.
It is not the model target. That choice waits for the closing note.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "data" / "raw" / "LoanData.csv"
FIGURE_DIR = REPO / "docs" / "bondora" / "figures"
COLUMNS = (
    "ReportAsOfEOD",
    "LoanDate",
    "Status",
    "DefaultDate",
    "Restructured",
    "CurrentDebtDaysPrimary",
    "DebtOccuredOn",
    "ActiveLateCategory",
)
DAYS_PER_MONTH = 30.437
STATUSES = ("Repaid", "Current", "Late")


def load_label_frame(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load status, default date, and the delinquency fields read beside them."""
    frame = pd.read_csv(csv_path, usecols=list(COLUMNS), engine="python", on_bad_lines="skip")
    for column in ("ReportAsOfEOD", "LoanDate", "DefaultDate", "DebtOccuredOn"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["year"] = frame["LoanDate"].dt.year
    frame["has_default_date"] = frame["DefaultDate"].notna()
    frame["months_observed"] = (frame["ReportAsOfEOD"] - frame["LoanDate"]).dt.total_seconds() / (
        86400 * DAYS_PER_MONTH
    )
    frame["months_to_default"] = (frame["DefaultDate"] - frame["LoanDate"]).dt.total_seconds() / (
        86400 * DAYS_PER_MONTH
    )
    frame["days_debt_to_default"] = (
        frame["DefaultDate"] - frame["DebtOccuredOn"]
    ).dt.total_seconds() / 86400
    observed_12m = frame["months_observed"] >= 12
    within_12m = frame["months_to_default"].between(0, 12, inclusive="both")
    frame["default_within_12m"] = frame["has_default_date"] & observed_12m & within_12m
    frame["eligible_12m"] = observed_12m & frame["LoanDate"].notna()
    return frame


def _status_cross(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(
        frame["Status"],
        frame["has_default_date"].map({False: "no_DefaultDate", True: "has_DefaultDate"}),
        margins=True,
    )
    return table.reindex(index=[*STATUSES, "All"])


def _months_by_status(frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.loc[frame["has_default_date"]]
    return dated.groupby("Status")["months_to_default"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )


def _year_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Ever-default on the vintage, and 12-month default on loans watched 12 months."""
    grouped = frame.groupby("year", dropna=True)
    ever = grouped["has_default_date"].mean()
    eligible = frame.loc[frame["eligible_12m"]]
    within = eligible.groupby("year")["default_within_12m"].mean()
    counts = pd.DataFrame(
        {
            "n": grouped.size(),
            "n_watched_12m": frame.groupby("year")["eligible_12m"].sum(),
            "ever_default_rate": ever,
            "default_within_12m_rate": within,
        }
    )
    return counts


def _late_without_date(frame: pd.DataFrame) -> pd.Series:
    late = frame.loc[frame["Status"].eq("Late") & ~frame["has_default_date"]]
    days = pd.to_numeric(late["CurrentDebtDaysPrimary"], errors="coerce")
    return pd.Series(
        {
            "n": len(late),
            "days_null": int(days.isna().sum()),
            "days_p50": float(days.median()),
            "days_p75": float(days.quantile(0.75)),
            "days_p90": float(days.quantile(0.9)),
            "share_ge_60": float((days >= 60).mean()),
            "share_ge_180": float((days >= 180).mean()),
        }
    )


def report(frame: pd.DataFrame) -> str:
    """Plain-text cross of Status and DefaultDate, plus the working 12-month mark."""
    dated = frame.loc[frame["has_default_date"]]
    gap = dated["days_debt_to_default"].dropna()
    gap_by_year = (
        dated.dropna(subset=["year"]).groupby("year")["days_debt_to_default"].median().round(1)
    )
    mature = frame.loc[frame["year"].between(2009, 2022) & frame["eligible_12m"]]
    mark = mature.loc[mature["default_within_12m"], "Status"].value_counts()
    current_dated = frame.loc[frame["Status"].eq("Current") & frame["has_default_date"]]
    late = frame.loc[frame["Status"].eq("Late") & ~frame["has_default_date"]]
    lines = [
        f"rows {len(frame)}",
        f"snapshot {frame['ReportAsOfEOD'].dropna().iloc[0]}",
        f"DefaultDate filled {int(frame['has_default_date'].sum())}",
        f"DefaultDate before LoanDate {int((dated['months_to_default'] < 0).sum())}",
        f"DefaultDate after snapshot {int((dated['DefaultDate'] > dated['ReportAsOfEOD']).sum())}",
        "",
        "Status x DefaultDate",
        _status_cross(frame).to_string(),
        "",
        "months from LoanDate to DefaultDate, by Status",
        _months_by_status(frame).round(2).to_string(),
        "",
        "share of dated defaults by horizon (months_to_default <= t, and >= 0)",
    ]
    positive = dated.loc[dated["months_to_default"] >= 0, "months_to_default"]
    for horizon in (3, 6, 12, 24, 36):
        share = float((positive <= horizon).mean())
        lines.append(f"  <= {horizon} months {share:.3f}")
    lines.extend(
        [
            "",
            "rates by origination year",
            _year_rates(frame).round(3).to_string(),
            "",
            "days from DebtOccuredOn to DefaultDate",
            f"dated loans missing DebtOccuredOn {int(dated['DebtOccuredOn'].isna().sum())}",
            f"share of gaps below 0 {float((gap < 0).mean()):.3f}",
            gap.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string(),
            "",
            "median days DebtOccuredOn to DefaultDate, by year",
            gap_by_year.to_string(),
            "",
            "Late and no DefaultDate, CurrentDebtDaysPrimary",
            _late_without_date(frame).round(3).to_string(),
            "",
            "ActiveLateCategory among Late with no DefaultDate",
            late["ActiveLateCategory"].value_counts(dropna=False).to_string(),
            "",
            "Restructured x has DefaultDate",
            pd.crosstab(frame["Restructured"], frame["has_default_date"], margins=True).to_string(),
            "",
            f"Current with DefaultDate {len(current_dated)}",
            current_dated["months_to_default"].describe().round(2).to_string(),
            "",
            "working mark on 2009-2022 loans watched >= 12 months",
            f"n {len(mature)}",
            f"default within 12 months {int(mature['default_within_12m'].sum())}",
            f"rate {float(mature['default_within_12m'].mean()):.3f}",
            "status mix of that mark",
            mark.to_string(),
        ]
    )
    return "\n".join(lines)


def write_figures(frame: pd.DataFrame, dest_dir: Path = FIGURE_DIR) -> list[Path]:
    """Status against DefaultDate, time to that date, and the 12-month rate by year."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cross = _status_cross(frame).drop(index="All")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = range(len(STATUSES))
    width = 0.36
    ax.bar(
        [p - width / 2 for p in positions],
        cross["no_DefaultDate"],
        width=width,
        label="No DefaultDate",
        color="#4c78a8",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        cross["has_DefaultDate"],
        width=width,
        label="DefaultDate set",
        color="#e45756",
    )
    ax.set_xticks(list(positions), list(STATUSES))
    ax.set_ylabel("Loans")
    ax.set_title("Status on the snapshot, with and without DefaultDate")
    ax.legend()
    fig.tight_layout()
    path = dest_dir / "status_x_default_date.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    dated = frame.loc[frame["has_default_date"] & frame["months_to_default"].between(0, 72)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(dated["months_to_default"], bins=36, color="#e45756")
    ax.axvline(12, color="#333333", linestyle="--", linewidth=1, label="12 months")
    ax.set_xlabel("Months from LoanDate to DefaultDate")
    ax.set_ylabel("Loans with a DefaultDate")
    ax.set_title("When collection started")
    ax.legend()
    fig.tight_layout()
    path = dest_dir / "months_to_default.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    rates = _year_rates(frame)
    rates = rates.loc[rates.index.to_series().between(2009, 2022)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        rates.index.astype(int),
        rates["ever_default_rate"],
        marker="o",
        color="#e45756",
        label="DefaultDate ever set",
    )
    ax.plot(
        rates.index.astype(int),
        rates["default_within_12m_rate"],
        marker="o",
        color="#4c78a8",
        label="DefaultDate within 12 months",
    )
    ax.set_ylabel("Share of the vintage")
    ax.set_title("Default mark by origination year, 2009–2022")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path = dest_dir / "default_rate_by_year.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    frame = load_label_frame()
    print(report(frame))
    for path in write_figures(frame):
        print(f"figure {path}")


if __name__ == "__main__":
    main()
