"""Bondora EDA step 1: what a row is, and the date of the snapshot.

Reads data/raw/LoanData.csv. Does not write a database or a model.
The C engine fails on this file when selecting columns, so this uses Python.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "LoanData.csv"
COLUMNS = (
    "ReportAsOfEOD",
    "LoanId",
    "LoanNumber",
    "PartyId",
    "LoanDate",
    "ListedOnUTC",
    "LoanApplicationStartedDate",
    "ContractEndDate",
    "MaturityDate_Original",
    "MaturityDate_Last",
    "Amount",
)


def load_grain_frame(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the identity and date columns used in EDA step 1."""
    frame = pd.read_csv(csv_path, usecols=list(COLUMNS), engine="python", on_bad_lines="skip")
    for column in (
        "ReportAsOfEOD",
        "LoanDate",
        "ListedOnUTC",
        "LoanApplicationStartedDate",
        "ContractEndDate",
        "MaturityDate_Original",
        "MaturityDate_Last",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0"
    return f"{100 * n / total:.1f}%"


def report(frame: pd.DataFrame) -> str:
    """Plain-text inventory of the step-1 columns."""
    n = len(frame)
    lines: list[str] = [f"rows {n}"]

    for column in COLUMNS:
        series = frame[column]
        nulls = int(series.isna().sum())
        n_unique = series.nunique(dropna=True)
        lines.append(f"{column}: nulls {nulls} ({_pct(nulls, n)}), nunique {n_unique}")

    snapshot = frame["ReportAsOfEOD"]
    lines.append(
        f"snapshot values {snapshot.nunique(dropna=True)} min {snapshot.min()} max {snapshot.max()}"
    )

    lines.append(f"duplicate LoanId {int(frame['LoanId'].duplicated().sum())}")
    lines.append(f"duplicate LoanNumber {int(frame['LoanNumber'].duplicated().sum())}")

    loans_per_party = frame.groupby("PartyId", dropna=False).size()
    lines.append(f"parties {len(loans_per_party)}")
    lines.append(f"parties with 1 loan {int((loans_per_party == 1).sum())}")
    lines.append(f"parties with 2+ loans {int((loans_per_party > 1).sum())}")
    lines.append(f"max loans per party {int(loans_per_party.max())}")

    issued = frame["LoanDate"].notna()
    lines.append(f"LoanDate min {frame['LoanDate'].min()} max {frame['LoanDate'].max()}")
    lines.append(f"rows without LoanDate {int((~issued).sum())}")

    loan_day = frame["LoanDate"].dt.floor("D")
    app_day = frame["LoanApplicationStartedDate"].dt.floor("D")
    list_day = frame["ListedOnUTC"].dt.floor("D")
    lines.append(f"application calendar before LoanDate {int((app_day < loan_day).sum())}")
    lines.append(f"application calendar same day {int((app_day == loan_day).sum())}")
    lines.append(f"application calendar after LoanDate {int((app_day > loan_day).sum())}")
    lines.append(f"listed calendar before LoanDate {int((list_day < loan_day).sum())}")
    lines.append(f"listed calendar same day {int((list_day == loan_day).sum())}")
    lines.append(f"listed calendar after LoanDate {int((list_day > loan_day).sum())}")

    end = frame["ContractEndDate"]
    snap = frame["ReportAsOfEOD"]
    ended = end.notna()
    lines.append(f"ContractEndDate filled {int(ended.sum())} ({_pct(int(ended.sum()), n)})")
    comparable = ended & snap.notna()
    lines.append(
        "contract end on or before snapshot "
        f"{int((end[comparable] <= snap[comparable]).sum())} of {int(comparable.sum())}"
    )
    lines.append(f"contract end after snapshot {int((end[comparable] > snap[comparable]).sum())}")

    original = frame["MaturityDate_Original"]
    last = frame["MaturityDate_Last"]
    both_mat = original.notna() & last.notna()
    lines.append(f"both maturities filled {int(both_mat.sum())}")
    delta_days = (last[both_mat] - original[both_mat]).dt.days
    lines.append(f"maturity unchanged {int((delta_days == 0).sum())}")
    lines.append(f"maturity extended {int((delta_days > 0).sum())}")
    lines.append(f"maturity extended more than 60 days {int((delta_days > 60).sum())}")
    lines.append(f"maturity extended more than 10 years {int((delta_days > 3650).sum())}")
    lines.append(f"maturity shortened {int((delta_days < 0).sum())}")

    amount = frame["Amount"]
    lines.append(f"Amount <= 0 {int((amount.fillna(0) <= 0).sum())}")
    lines.append(f"Amount null {int(amount.isna().sum())}")
    return "\n".join(lines)


FIGURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "bondora"
    / "figures"
    / "application_vs_loan_date.png"
)


def calendar_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Calendar-day order of application and listing relative to LoanDate.

    LoanDate is stored at midnight, so a timestamp comparison is not this.
    """
    loan_day = frame["LoanDate"].dt.floor("D")
    counts: dict[str, dict[str, int]] = {}
    for name, column in (
        ("Application started", "LoanApplicationStartedDate"),
        ("Listed", "ListedOnUTC"),
    ):
        day = frame[column].dt.floor("D")
        counts[name] = {
            "before": int((day < loan_day).sum()),
            "same day": int((day == loan_day).sum()),
            "after": int((day > loan_day).sum()),
        }
    return counts


def write_calendar_figure(frame: pd.DataFrame, dest: Path = FIGURE_PATH) -> Path:
    """Bar chart of calendar-day order. Written next to the public note."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = calendar_counts(frame)
    labels = ["before LoanDate", "same calendar day", "after LoanDate"]
    keys = ["before", "same day", "after"]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, name in ((-width / 2, "Application started"), (width / 2, "Listed")):
        values = [counts[name][key] for key in keys]
        ax.bar([i + offset for i in x], values, width=width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Loans")
    ax.set_title("When the application happens, relative to the issue date")
    ax.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def main() -> None:
    frame = load_grain_frame()
    print(report(frame))
    path = write_calendar_figure(frame)
    print(f"figure {path}")


if __name__ == "__main__":
    main()
