"""Data layer: ingest once, query via SQL thereafter."""

from src.data.queries import count_by_credit_class, fetch_loan_by_id, fetch_loans

__all__ = ["count_by_credit_class", "fetch_loan_by_id", "fetch_loans"]
