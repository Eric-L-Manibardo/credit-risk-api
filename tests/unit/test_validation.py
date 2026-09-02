"""Validation is pure DataFrame logic: no engine, no CSV, no network."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import DataValidationError, collect_issues, validate_loans
from tests.unit.conftest import VALID_LOAN


def test_valid_frame_has_no_issues(valid_loans_frame: pd.DataFrame) -> None:
    assert collect_issues(valid_loans_frame) == []
    validate_loans(valid_loans_frame)


def test_age_above_sample_max_is_allowed(valid_loans_frame: pd.DataFrame) -> None:
    """Domain range is [18, 100], not the German Credit sample max of 75."""
    valid_loans_frame.loc[0, "age"] = 76
    assert collect_issues(valid_loans_frame) == []


def test_empty_frame_is_invalid() -> None:
    issues = collect_issues(pd.DataFrame())
    assert issues == ["frame is empty"]


def test_collects_all_issues_not_just_the_first(valid_loans_frame: pd.DataFrame) -> None:
    valid_loans_frame.loc[0, "age"] = 12
    valid_loans_frame.loc[0, "credit_class"] = "maybe"
    valid_loans_frame.loc[1, "checking_status"] = None
    issues = collect_issues(valid_loans_frame)
    joined = " ".join(issues)
    assert "age" in joined
    assert "credit_class" in joined
    assert "checking_status" in joined
    assert len(issues) >= 3


def test_validate_loans_raises_with_issue_list(valid_loans_frame: pd.DataFrame) -> None:
    valid_loans_frame.loc[0, "age"] = 12
    with pytest.raises(DataValidationError) as exc_info:
        validate_loans(valid_loans_frame)
    assert any("age" in issue for issue in exc_info.value.issues)


def test_duplicate_ids_are_flagged() -> None:
    row = dict(VALID_LOAN)
    df = pd.DataFrame([{**row, "id": 1}, {**row, "id": 1, "age": 40}])
    issues = collect_issues(df)
    assert any("duplicate" in issue for issue in issues)


def test_empty_string_is_not_a_valid_category(valid_loans_frame: pd.DataFrame) -> None:
    valid_loans_frame.loc[0, "housing"] = "   "
    issues = collect_issues(valid_loans_frame)
    assert any("housing" in issue and "empty" in issue for issue in issues)
