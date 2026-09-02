"""Download the German Credit dataset into data/raw/.

This is the only place that talks to the internet for data. Ingest and
runtime code read the local CSV or SQLite, never this URL.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# OpenML dataset 31 ("credit-g"): UCI Statlog German Credit with readable
# column names instead of the original A11/A12 codes.
DATASET_URL = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_CSV_PATH = RAW_DIR / "german_credit.csv"


def download_dataset(*, force: bool = False, dest: Path | None = None) -> Path:
    """Download the dataset CSV if it is missing (or if force=True).

    Returns the path to the local CSV.
    """
    dest = dest or RAW_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    urllib.request.urlretrieve(DATASET_URL, dest)
    return dest


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset ready at {path}")
