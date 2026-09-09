"""Sealed stratified test split and 5-fold CV on the remainder."""

from __future__ import annotations

from typing import cast

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.features.engineering import CATEGORICAL_FEATURES
from src.model.config import N_FOLDS, RANDOM_STATE, TEST_SIZE

TARGET_COL = "credit_class"
ID_COL = "id"

# x_rest, x_test, y_rest, y_test, ids_rest, ids_test
SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]


def frame_to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sort by id, drop id/target from X, encode bad=1. Same order every run."""
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"frame must include {ID_COL!r} and {TARGET_COL!r}")
    ordered = df.sort_values(ID_COL).reset_index(drop=True)
    y = (ordered[TARGET_COL] == "bad").astype(int)
    ids = ordered[ID_COL]
    x = ordered.drop(columns=[ID_COL, TARGET_COL])
    missing_cats = [col for col in CATEGORICAL_FEATURES if col not in x.columns]
    if missing_cats:
        raise ValueError(f"X missing categorical columns: {missing_cats}")
    return x, y, ids


def sealed_test_split(x: pd.DataFrame, y: pd.Series, ids: pd.Series) -> SplitResult:
    """Stratified 15% test. Remainder is the CV pool. Seed is RANDOM_STATE."""
    split = train_test_split(
        x,
        y,
        ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return cast(SplitResult, split)


def cv_folds(y_rest: pd.Series) -> StratifiedKFold:
    if len(y_rest) < N_FOLDS:
        raise ValueError(f"need at least {N_FOLDS} rows for CV, got {len(y_rest)}")
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
