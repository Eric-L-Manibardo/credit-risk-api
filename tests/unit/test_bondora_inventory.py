"""Bondora step 5: every column in one group, and what counts as not recorded."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "EDA_bondora" / "bondora_step5_inventory.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bondora_step5_inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _load()


def _all_columns() -> list[str]:
    return [column for columns in inventory.GROUPS.values() for column in columns]


def test_groups_cover_the_112_columns_once() -> None:
    columns = _all_columns()
    assert len(columns) == 112
    assert len(set(columns)) == 112


def test_flagged_columns_are_origination() -> None:
    assert set(inventory.FLAGGED) <= set(inventory.GROUPS["origination"])


def test_check_groups_accepts_the_matching_header() -> None:
    inventory.check_groups(_all_columns())


def test_check_groups_rejects_an_unassigned_column() -> None:
    with pytest.raises(ValueError, match=r"unassigned \['NewColumn'\]"):
        inventory.check_groups([*_all_columns(), "NewColumn"])


def test_check_groups_rejects_a_column_missing_from_the_file() -> None:
    header = _all_columns()
    with pytest.raises(ValueError, match=rf"not in file \['{header[0]}'\]"):
        inventory.check_groups(header[1:])


def _toy_frame(rows: Sequence[Mapping[str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(inventory.MEASURED))


def test_not_recorded_reads_codes_placeholders_and_zeros() -> None:
    filled = dict.fromkeys(inventory.MEASURED, "1")
    coded = {
        **filled,
        "UseOfLoan": "-1",
        "CreditScoreEsMicroL": "M",
        "DebtToIncome": ".0000",
        **dict.fromkeys(inventory.INCOME_BREAKDOWN, ".00"),
        "Age": None,
    }
    missing = inventory.not_recorded(_toy_frame([filled, coded]))
    assert not missing.iloc[0].any()
    for column in ("UseOfLoan", "CreditScoreEsMicroL", "DebtToIncome", "Age"):
        assert missing.loc[1, column]
    assert missing.loc[1, list(inventory.INCOME_BREAKDOWN)].all()
    assert not missing.loc[1, "IncomeTotal"]


def test_an_empty_repayment_counts_only_when_there_were_previous_loans() -> None:
    first_loan: dict[str, str | None] = dict.fromkeys(inventory.MEASURED, "1")
    first_loan.update(NoOfPreviousLoansBeforeLoan="0", PreviousRepaymentsBeforeLoan=None)
    returning = dict(first_loan, NoOfPreviousLoansBeforeLoan="2")
    missing = inventory.not_recorded(_toy_frame([first_loan, returning]))
    assert list(missing["PreviousRepaymentsBeforeLoan"]) == [False, True]


def test_a_single_zero_income_part_is_recorded() -> None:
    row = dict.fromkeys(inventory.MEASURED, "1")
    row["IncomeFromPension"] = ".00"
    missing = inventory.not_recorded(_toy_frame([row]))
    assert not missing.loc[0, "IncomeFromPension"]
