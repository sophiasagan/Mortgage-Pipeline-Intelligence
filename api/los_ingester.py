"""
los_ingester.py
===============
Parses a LOS (Loan Origination System) export — either a CSV file or a raw
list of dicts — into a normalised pandas DataFrame ready for risk_explainer
and action_generator.

Adapt the COLUMN_MAP for your LOS export format:
  Encompass:        loan_number, milestone, appStatus, titleStatus, ...
  BytePro:          LoanNumber, Stage, AppraisalStatus, ...
  MortgageDirector: loan_id, pipeline_stage, ...

Usage:
    from api.los_ingester import ingest_csv, ingest_records
    df = ingest_csv("data/raw/pipeline_export.csv")
    df = ingest_records([{"loan_id": "LN-1", ...}, ...])
"""

from __future__ import annotations

import pathlib
import logging
from typing import Any

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected canonical column names (matches train_pipeline_model.py features)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "loan_id",
    "borrower_initials",
    "loan_amount",
    "pipeline_stage",
    "days_in_current_stage",
    "days_since_last_status_change",
    "days_to_projected_close",
    "appraisal_status",
    "title_status",
    "income_docs_complete",
    "rate_lock_expiry_days",
    "condition_count",
    "prior_fall_out_same_stage",
    "loan_type",
    "ltv",
    "dti",
    "credit_score_tier",
]

# ---------------------------------------------------------------------------
# Column name mapping — update the LEFT side for your LOS export headers
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    # ── generic / MortgageDirector ──────────────────────────────────────────
    "loan_id":                       "loan_id",
    "borrower_initials":             "borrower_initials",
    "loan_amount":                   "loan_amount",
    "pipeline_stage":                "pipeline_stage",
    "days_in_current_stage":         "days_in_current_stage",
    "days_since_last_status_change": "days_since_last_status_change",
    "days_to_projected_close":       "days_to_projected_close",
    "appraisal_status":              "appraisal_status",
    "title_status":                  "title_status",
    "income_docs_complete":          "income_docs_complete",
    "rate_lock_expiry_days":         "rate_lock_expiry_days",
    "condition_count":               "condition_count",
    "prior_fall_out_same_stage":     "prior_fall_out_same_stage",
    "loan_type":                     "loan_type",
    "ltv":                           "ltv",
    "dti":                           "dti",
    "credit_score_tier":             "credit_score_tier",

    # ── Encompass aliases ───────────────────────────────────────────────────
    "loan_number":    "loan_id",
    "milestone":      "pipeline_stage",
    "appstatus":      "appraisal_status",
    "titlestatus":    "title_status",
    "loanamount":     "loan_amount",
    "loantype":       "loan_type",
    "ratelock":       "rate_lock_expiry_days",
    "openconditions": "condition_count",

    # ── BytePro aliases ─────────────────────────────────────────────────────
    "loannumber": "loan_id",
    "stage":      "pipeline_stage",
}

# ---------------------------------------------------------------------------
# Stage normalisation — map LOS milestone names to canonical stage values
# ---------------------------------------------------------------------------
STAGE_MAP: dict[str, str] = {
    # canonical
    "application":    "application",
    "processing":     "processing",
    "underwriting":   "underwriting",
    "cond_approval":  "cond_approval",
    "clear_to_close": "clear_to_close",
    # Encompass milestones
    "app submitted":         "application",
    "application":           "application",
    "proc":                  "processing",
    "processing":            "processing",
    "uw":                    "underwriting",
    "underwriting":          "underwriting",
    "conditional approval":  "cond_approval",
    "cond. approval":        "cond_approval",
    "ctc":                   "clear_to_close",
    "clear to close":        "clear_to_close",
}

# ---------------------------------------------------------------------------
# Column default values (used when a field is missing from the export)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "borrower_initials":             "X.X.",
    "loan_amount":                   300_000,
    "days_in_current_stage":         7,
    "days_since_last_status_change": 3,
    "days_to_projected_close":       30,
    "appraisal_status":              "ordered",
    "title_status":                  "ordered",
    "income_docs_complete":          0,
    "rate_lock_expiry_days":         30,
    "condition_count":               2,
    "prior_fall_out_same_stage":     0,
    "loan_type":                     "conventional",
    "ltv":                           80.0,
    "dti":                           38.0,
    "credit_score_tier":             "good",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fill defaults, coerce types."""
    # Lowercase all column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Rename to canonical names
    df = df.rename(columns={k.lower(): v for k, v in COLUMN_MAP.items()})

    # Fill missing canonical columns with defaults
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            log.warning("Column '%s' not in export — filling with default: %s", col, default)
            df[col] = default

    # Normalise pipeline_stage
    if "pipeline_stage" in df.columns:
        df["pipeline_stage"] = (
            df["pipeline_stage"]
            .str.lower()
            .str.strip()
            .map(lambda s: STAGE_MAP.get(s, s))
        )

    # Coerce numeric columns
    for col in ["loan_amount", "days_in_current_stage", "days_since_last_status_change",
                "days_to_projected_close", "rate_lock_expiry_days", "condition_count",
                "ltv", "dti"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS.get(col, 0))

    # Coerce boolean columns
    for col in ["income_docs_complete", "prior_fall_out_same_stage"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: 1 if str(v).lower() in ("1", "true", "yes", "y") else 0
            )

    # Ensure loan_id is string
    if "loan_id" in df.columns:
        df["loan_id"] = df["loan_id"].astype(str)

    return df[CANONICAL_COLUMNS]


def ingest_csv(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a LOS CSV export and return a normalised DataFrame."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LOS export not found: {path}")
    raw = pd.read_csv(path, dtype=str)
    log.info("Loaded %d rows from %s", len(raw), path)
    return _normalise(raw)


def ingest_records(records: list[dict]) -> pd.DataFrame:
    """Convert a list of dicts (e.g. from an API response) to a normalised DataFrame."""
    if not records:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    raw = pd.DataFrame(records)
    return _normalise(raw)
