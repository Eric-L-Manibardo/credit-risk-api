"""Bondora EDA step 5: which columns existed on the day the loan was granted?

Reads data/raw/LoanData.csv. Does not write a database or a model.
The C engine fails on this file when selecting columns, so this uses Python.

Every column of the extract sits in exactly one group. A column whose timing
is unclear goes to "after_disbursement" until shown otherwise. The script
stops if the CSV header and the groups disagree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "data" / "raw" / "LoanData.csv"
FIGURE_DIR = REPO / "docs" / "bondora" / "figures"
COMPARE = ("EE", "FI", "ES")
COUNTRY_ORDER = ("EE", "FI", "ES", "NL", "SK")
PERIODS = {"2014-2016": (2014, 2016), "2018-2022": (2018, 2022)}

INCOME_BREAKDOWN = (
    "IncomeFromPrincipalEmployer",
    "IncomeFromPension",
    "IncomeFromFamilyAllowance",
    "IncomeFromSocialWelfare",
    "IncomeFromLeavePay",
    "IncomeFromChildSupport",
    "IncomeOther",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "origination": (
        "NewCreditCustomer",
        "ApplicationSignedHour",
        "ApplicationSignedWeekday",
        "VerificationType",
        "LanguageCode",
        "Age",
        "Gender",
        "Country",
        "AppliedAmount",
        "Amount",
        "Interest",
        "LoanDuration",
        "MonthlyPayment",
        "MonthlyPaymentDay",
        "UseOfLoan",
        "Education",
        "MaritalStatus",
        "NrOfDependants",
        "EmploymentStatus",
        "EmploymentDurationCurrentEmployer",
        "WorkExperience",
        "OccupationArea",
        "HomeOwnershipType",
        *INCOME_BREAKDOWN,
        "IncomeTotal",
        "ExistingLiabilities",
        "LiabilitiesTotal",
        "RefinanceLiabilities",
        "DebtToIncome",
        "FreeCash",
        "NoOfPreviousLoansBeforeLoan",
        "AmountOfPreviousLoansBeforeLoan",
        "PreviousRepaymentsBeforeLoan",
        "PreviousEarlyRepaymentsBefoleLoan",
        "PreviousEarlyRepaymentsCountBeforeLoan",
    ),
    "one_country": (
        "CreditScoreEsMicroL",
        "CreditScoreEsEquifaxRisk",
        "CreditScoreFiAsiakasTietoRiskGrade",
        "CreditScoreEeMini",
    ),
    "platform": (
        "ModelVersion",
        "Rating",
        "ProbabilityOfDefault",
        "ExpectedLoss",
        "ExpectedReturn",
        "LossGivenDefault",
        "EL_V0",
        "Rating_V0",
        "EL_V1",
        "Rating_V1",
        "Rating_V2",
        "BidsPortfolioManager",
        "BidsApi",
        "BidsManual",
    ),
    "after_disbursement": (
        "Status",
        "DefaultDate",
        "Restructured",
        "ContractEndDate",
        "MaturityDate_Last",
        "ActiveScheduleFirstPaymentReached",
        "PlannedPrincipalTillDate",
        "PlannedInterestTillDate",
        "LastPaymentOn",
        "CurrentDebtDaysPrimary",
        "DebtOccuredOn",
        "CurrentDebtDaysSecondary",
        "DebtOccuredOnForSecondary",
        "PrincipalOverdueBySchedule",
        "PlannedPrincipalPostDefault",
        "PlannedInterestPostDefault",
        "EAD1",
        "EAD2",
        "PrincipalRecovery",
        "InterestRecovery",
        "RecoveryStage",
        "StageActiveSince",
        "ActiveLateCategory",
        "WorseLateCategory",
        "ActiveLateLastPaymentCategory",
        "PrincipalPaymentsMade",
        "InterestAndPenaltyPaymentsMade",
        "PrincipalWriteOffs",
        "InterestAndPenaltyWriteOffs",
        "PrincipalBalance",
        "InterestAndPenaltyBalance",
        "PrincipalDebtServicingCost",
        "InterestAndPenaltyDebtServicingCost",
        "GracePeriodStart",
        "GracePeriodEnd",
        "NextPaymentDate",
        "NextPaymentNr",
        "NrOfScheduledPayments",
        "ReScheduledOn",
    ),
    "reference": (
        "ReportAsOfEOD",
        "LoanId",
        "LoanNumber",
        "PartyId",
        "LoanApplicationStartedDate",
        "ListedOnUTC",
        "BiddingStartedOn",
        "LoanDate",
        "FirstPaymentDate",
        "MaturityDate_Original",
    ),
    "empty": ("DateOfBirth", "County", "City", "EmploymentPosition"),
}

# Origination columns that are known on the day but carry Bondora's own pricing.
FLAGGED = {
    "Interest": "set by Bondora from its rating",
    "MonthlyPayment": "follows from Interest",
}

CODED_MISSING = {
    "UseOfLoan": ("-1",),
    "Education": ("-1",),
    "MaritalStatus": ("-1",),
    "EmploymentStatus": ("-1",),
    "OccupationArea": ("-1",),
    "HomeOwnershipType": ("-1",),
    "CreditScoreEsMicroL": ("M",),
}
ZERO_MEANS_MISSING = ("DebtToIncome", "FreeCash")
# From 2018 an empty amount means zero, unless the count beside it says there was something.
EMPTY_MEANS_NONE = {
    "PreviousRepaymentsBeforeLoan": "NoOfPreviousLoansBeforeLoan",
    "PreviousEarlyRepaymentsBefoleLoan": "PreviousEarlyRepaymentsCountBeforeLoan",
}
MEASURED = GROUPS["origination"] + GROUPS["one_country"]
BIDS = ("BidsPortfolioManager", "BidsApi", "BidsManual")
CONTEXT = ("NrOfScheduledPayments", *BIDS)
SCORE_COUNTRY = {
    "CreditScoreEsMicroL": "ES",
    "CreditScoreEsEquifaxRisk": "ES",
    "CreditScoreFiAsiakasTietoRiskGrade": "FI",
    "CreditScoreEeMini": "EE",
}
CUT_DAY = "2017-06-01"


def check_groups(header: list[str]) -> None:
    """Raise if a column is in two groups, in none, or not in the file."""
    assigned = [column for columns in GROUPS.values() for column in columns]
    repeated = sorted({column for column in assigned if assigned.count(column) > 1})
    unassigned = sorted(set(header) - set(assigned))
    unknown = sorted(set(assigned) - set(header))
    if repeated or unassigned or unknown:
        raise ValueError(f"repeated {repeated}, unassigned {unassigned}, not in file {unknown}")


def load_inventory_frame(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Origination and one-country columns as text, plus year and the context columns."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    check_groups(header)
    columns = ["LoanDate", *MEASURED, *CONTEXT]
    frame = pd.read_csv(csv_path, usecols=columns, dtype=str, engine="python", on_bad_lines="skip")
    frame["year"] = pd.to_datetime(frame["LoanDate"], errors="coerce").dt.year
    return frame


def not_recorded(frame: pd.DataFrame) -> pd.DataFrame:
    """True where a measured column carries no information for that loan.

    Empty cells, the -1 codes, the "M" placeholder of the Spanish score, a zero
    debt-to-income or free cash, and an income breakdown whose parts add to zero.
    An empty previous-repayment amount counts only when its count is above zero.
    """
    missing = frame[list(MEASURED)].isna()
    for column, count in EMPTY_MEANS_NONE.items():
        missing[column] &= pd.to_numeric(frame[count], errors="coerce").gt(0)
    for column, codes in CODED_MISSING.items():
        missing[column] |= frame[column].isin(codes)
    for column in ZERO_MEANS_MISSING:
        missing[column] |= pd.to_numeric(frame[column], errors="coerce").eq(0)
    parts = frame[list(INCOME_BREAKDOWN)].apply(pd.to_numeric, errors="coerce")
    breakdown_empty = parts.fillna(0).sum(axis=1).eq(0)
    for column in INCOME_BREAKDOWN:
        missing[column] |= breakdown_empty
    return missing


def by_year(missing: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    """Share not recorded, one row per measured column, one column per year."""
    return missing.groupby(years).mean().T


def by_country_period(missing: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Share not recorded for EE, FI, ES, before and after the June 2017 cut."""
    blocks: dict[str, pd.Series] = {}
    for label, (start, end) in PERIODS.items():
        for country in COMPARE:
            rows = frame["year"].between(start, end) & frame["Country"].eq(country)
            blocks[f"{country} {label}"] = missing.loc[rows].mean()
    return pd.DataFrame(blocks)


def publication_cut(frame: pd.DataFrame) -> tuple[str, str]:
    """Last LoanDate with a UseOfLoan code, first LoanDate with -1."""
    coded = frame["UseOfLoan"].ne("-1") & frame["UseOfLoan"].notna()
    return str(frame.loc[coded, "LoanDate"].max()), str(frame.loc[~coded, "LoanDate"].min())


def score_fill_by_year(missing: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Share of each country's loans with its own bureau score, by year."""
    rows = {}
    for column, country in SCORE_COUNTRY.items():
        own = frame["Country"].eq(country)
        rows[f"{column} ({country})"] = (~missing.loc[own, column]).groupby(frame["year"]).mean()
    return pd.DataFrame(rows).T


def _context(frame: pd.DataFrame) -> list[str]:
    """Why the bids and the schedule count sit outside origination."""
    amount = pd.to_numeric(frame["Amount"], errors="coerce")
    bids = frame[list(BIDS)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    scheduled = frame["NrOfScheduledPayments"]
    matches = scheduled[scheduled.notna()].eq(frame.loc[scheduled.notna(), "LoanDuration"])
    return [
        "median share of Amount covered by bids, by year",
        (bids / amount).groupby(frame["year"]).median().round(2).to_string(),
        "",
        f"NrOfScheduledPayments equals LoanDuration on {matches.mean():.3f} of filled rows",
    ]


def report(frame: pd.DataFrame) -> str:
    """Groups, the June 2017 cut, and where each measured column is filled."""
    missing = not_recorded(frame)
    last_coded, first_uncoded = publication_cut(frame)
    previous = pd.to_numeric(frame["NoOfPreviousLoansBeforeLoan"], errors="coerce").gt(0)
    repayments_empty = frame["PreviousRepaymentsBeforeLoan"].isna()
    recent = frame["year"].ge(2018)
    mature = frame["year"].between(2009, 2022)
    after_cut = frame["LoanDate"].ge(CUT_DAY)
    full_window = frame["year"].between(*PERIODS["2014-2016"])
    lines = [f"rows {len(frame)}", ""]
    for group, columns in GROUPS.items():
        lines.append(f"{group} ({len(columns)}): {', '.join(columns)}")
    lines += [
        "",
        "flagged origination columns",
        *(f"  {column}: {reason}" for column, reason in FLAGGED.items()),
        "",
        f"UseOfLoan last coded {last_coded}, first -1 {first_uncoded}",
        f"2009-2022 loans {int(mature.sum())}, from {CUT_DAY} {int((mature & after_cut).sum())}",
        "",
        "loans 2014-2016 by country",
        pd.crosstab(frame.loc[full_window, "year"], frame.loc[full_window, "Country"])
        .reindex(columns=list(COMPARE))
        .to_string(),
        "",
        *_context(frame),
        "",
        "share not recorded, by origination year",
        by_year(missing, frame["year"]).round(2).to_string(),
        "",
        "share not recorded, by country and period",
        by_country_period(missing, frame).round(2).to_string(),
        "",
        "one-country scores filled, by country",
        (~missing[list(GROUPS["one_country"])])
        .groupby(frame["Country"])
        .mean()
        .reindex(list(COUNTRY_ORDER))
        .round(3)
        .T.to_string(),
        "",
        "own-country score filled, by year",
        score_fill_by_year(missing, frame).round(2).to_string(),
        "",
        "PreviousRepaymentsBeforeLoan empty, 2018-2024",
        f"  with previous loans {repayments_empty[previous & recent].mean():.3f}",
        f"  without previous loans {repayments_empty[~previous & recent].mean():.3f}",
    ]
    return "\n".join(lines)


def _heatmap(table: pd.DataFrame, title: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = table.to_numpy(dtype=float)
    height = 1.2 + 0.22 * len(table.index)
    width = max(7.5, 2.6 + 0.48 * len(table.columns))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(grid, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns], rotation=45)
    ax.set_yticks(range(len(table.index)), list(table.index), fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Share not recorded")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_figures(frame: pd.DataFrame, dest_dir: Path = FIGURE_DIR) -> list[Path]:
    """Not-recorded share by year, and by country before and after the cut."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = not_recorded(frame)
    origination = list(GROUPS["origination"])
    yearly = by_year(missing[origination], frame["year"])
    yearly.columns = [int(year) for year in yearly.columns]
    return [
        _heatmap(
            yearly,
            "Origination columns not recorded, by year",
            dest_dir / "origination_missing_by_year.png",
        ),
        _heatmap(
            by_country_period(missing, frame),
            "Not recorded, by country and period",
            dest_dir / "origination_missing_by_country.png",
        ),
    ]


def main() -> None:
    frame = load_inventory_frame()
    print(report(frame))
    for path in write_figures(frame):
        print(f"figure {path}")


if __name__ == "__main__":
    main()
