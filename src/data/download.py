"""Download loan extracts into data/raw/.

This is the only place that talks to the internet for data. Ingest and
runtime code read the local CSV or SQLite, never these URLs.

Kaggle credentials stay outside the repo: ``~/.kaggle/access_token`` or the
``KAGGLE_API_TOKEN`` environment variable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# OpenML dataset 31 ("credit-g"): UCI Statlog German Credit with readable
# column names instead of the original A11/A12 codes.
DATASET_URL = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff"
# Public Bondora loan book snapshot (EUR, originated 2009–2024).
BONDORA_DATASET = "dumbstatistician/loan-data-bondora"
BONDORA_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{BONDORA_DATASET}"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_CSV_PATH = RAW_DIR / "german_credit.csv"
BONDORA_CSV_PATH = RAW_DIR / "LoanData.csv"


def download_dataset(*, force: bool = False, dest: Path | None = None) -> Path:
    """Download the German Credit CSV if it is missing (or if force=True).

    Returns the path to the local CSV.
    """
    dest = dest or RAW_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    urllib.request.urlretrieve(DATASET_URL, dest)
    return dest


def _kaggle_token_instructions() -> str:
    """Tell the caller how to create a personal Kaggle token. Never includes one."""
    return (
        "Create your own token at https://www.kaggle.com/settings/api "
        "(Generate New Token). It starts with KGAT_ and only works for your account.\n"
        "Save it outside the repo:\n"
        "  mkdir -p ~/.kaggle\n"
        "  printf '%s\\n' 'YOUR_TOKEN' > ~/.kaggle/access_token\n"
        "  chmod 600 ~/.kaggle/access_token\n"
        "Or set KAGGLE_API_TOKEN. Do not commit the token."
    )


def _kaggle_token() -> str:
    """Read a KGAT token from the environment or ~/.kaggle/access_token."""
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token:
        return env_token

    token_path = Path.home() / ".kaggle" / "access_token"
    if not token_path.is_file():
        raise FileNotFoundError(
            "Kaggle token missing. Bondora is not public without one.\n"
            + _kaggle_token_instructions()
        )
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise FileNotFoundError(
            f"Kaggle token file is empty: {token_path}\n" + _kaggle_token_instructions()
        )
    return token


def _extract_bondora_csv(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        preferred = [name for name in csv_members if Path(name).name == "LoanData.csv"]
        if not preferred and not csv_members:
            raise FileNotFoundError(f"Kaggle zip has no CSV: {archive.namelist()}")
        member = (preferred or csv_members)[0]
        with archive.open(member) as source, dest.open("wb") as target:
            shutil.copyfileobj(source, target)


def download_bondora(*, force: bool = False, dest: Path | None = None) -> Path:
    """Download the Bondora loan CSV from Kaggle if it is missing (or if force=True).

    Returns the path to the local CSV. Does not touch the German Credit extract.
    """
    dest = dest or BONDORA_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    request = urllib.request.Request(
        BONDORA_DOWNLOAD_URL,
        headers={"Authorization": f"Bearer {_kaggle_token()}"},
    )
    partial = dest.with_suffix(".csv.partial")
    try:
        with (
            urllib.request.urlopen(request) as response,
            tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".zip", delete=False) as tmp,
        ):
            zip_path = Path(tmp.name)
            shutil.copyfileobj(response, tmp)
        _extract_bondora_csv(zip_path, partial)
        partial.replace(dest)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"Kaggle rejected the token (HTTP {exc.code}).\n" + _kaggle_token_instructions()
            ) from exc
        raise RuntimeError(f"Kaggle download failed with HTTP {exc.code}") from exc
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if "zip_path" in locals():
            zip_path.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a loan extract into data/raw/.")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=("german", "bondora"),
        default="german",
        help="Which extract to fetch (default: german)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the CSV exists")
    args = parser.parse_args()

    if args.dataset == "bondora":
        path = download_bondora(force=args.force)
    else:
        path = download_dataset(force=args.force)
    print(f"Dataset ready at {path}")
