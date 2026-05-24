"""
risk_explainer.py
=================
Loads the trained XGBoost pipeline model and provides:

    explain_loan(loan_features: dict) -> dict

Returns close probability, risk level, and the top-3 risk factors with
SHAP-derived impact scores and plain-English labels.

Usage (standalone):
    python api/risk_explainer.py
"""

import pickle
import pathlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "model" / "pipeline_model.pkl"

# ---------------------------------------------------------------------------
# Risk-level thresholds (aligned with CLAUDE.md at-risk threshold of 0.65)
# ---------------------------------------------------------------------------
RISK_LEVELS = [
    (0.80, "low"),        # close_prob >= 0.80 → low risk
    (0.65, "moderate"),   # 0.65 ≤ close_prob < 0.80
    (0.40, "high"),       # 0.40 ≤ close_prob < 0.65
    (0.00, "critical"),   # close_prob < 0.40
]

# ---------------------------------------------------------------------------
# Plain-English label templates
# Each callable receives the feature value and the full loan_features dict
# ---------------------------------------------------------------------------

def _stage_label(val: Any, feat: dict) -> str:
    stage = feat.get("pipeline_stage", "current stage")
    days  = int(feat.get("days_in_current_stage", val))
    return f"Loan has been in {stage} for {days} days — above average"


def _days_no_change_label(val: Any, feat: dict) -> str:
    days = int(val)
    return f"No status change in {days} days — file may be stalling"


def _days_to_close_label(val: Any, feat: dict) -> str:
    days = int(val)
    return f"Only {days} days until projected close — limited time to resolve issues"


def _appraisal_label(val: Any, feat: dict) -> str:
    status_map = {
        "disputed": "Appraisal is currently disputed — can delay or kill the deal",
        "ordered":  "Appraisal ordered but not yet received",
        "received": "Appraisal received and on file",
        "waived":   "Appraisal waived — no appraisal risk",
    }
    raw_status = feat.get("appraisal_status", str(val))
    return status_map.get(raw_status, f"Appraisal status: {raw_status}")


def _title_label(val: Any, feat: dict) -> str:
    status_map = {
        "issue":   "Title has an open issue — must be resolved before closing",
        "ordered": "Title ordered but not yet cleared",
        "cleared": "Title cleared — no title risk",
    }
    raw_status = feat.get("title_status", str(val))
    return status_map.get(raw_status, f"Title status: {raw_status}")


def _income_docs_label(val: Any, feat: dict) -> str:
    complete = bool(int(val)) if val is not None else feat.get("income_docs_complete", False)
    if complete:
        return "Income documentation is complete"
    return "Income documentation incomplete — required for underwriting sign-off"


def _rate_lock_label(val: Any, feat: dict) -> str:
    days = int(val)
    if days <= 0:
        return "Rate lock has already expired — must be extended immediately"
    if days <= 7:
        return f"Rate lock expires in {days} days — extremely tight timeline"
    if days <= 14:
        return f"Rate lock expires in {days} days — tight timeline"
    return f"Rate lock expires in {days} days — monitor closely"


def _condition_count_label(val: Any, feat: dict) -> str:
    n = int(round(float(val)))
    if n == 0:
        return "No outstanding conditions — file is clean"
    if n == 1:
        return "1 outstanding condition not yet cleared"
    return f"{n} outstanding conditions not yet cleared"


def _prior_fallout_label(val: Any, feat: dict) -> str:
    flag = bool(int(val)) if val is not None else feat.get("prior_fall_out_same_stage", False)
    if flag:
        stage = feat.get("pipeline_stage", "this stage")
        return f"Prior fall-out recorded at {stage} — elevated recurrence risk"
    return "No prior fall-out history at this stage"


def _loan_type_label(val: Any, feat: dict) -> str:
    loan_type = feat.get("loan_type", str(val))
    notes = {
        "USDA": "USDA loans have longer government review timelines",
        "VA":   "VA loans require additional veteran eligibility checks",
        "FHA":  "FHA loans require MIP and strict property standards",
        "conventional": "Conventional loan — standard process",
    }
    return notes.get(loan_type, f"Loan type: {loan_type}")


def _ltv_label(val: Any, feat: dict) -> str:
    ltv = float(val)
    if ltv >= 95:
        return f"LTV of {ltv:.0f}% is very high — PMI required and elevated risk"
    if ltv >= 90:
        return f"LTV of {ltv:.0f}% is high — PMI required"
    return f"LTV of {ltv:.0f}% — within acceptable range"


def _dti_label(val: Any, feat: dict) -> str:
    dti = float(val)
    if dti >= 50:
        return f"DTI of {dti:.0f}% is critically high — may exceed guideline limits"
    if dti >= 45:
        return f"DTI of {dti:.0f}% is elevated — above typical guideline thresholds"
    return f"DTI of {dti:.0f}% — within guideline range"


def _credit_tier_label(val: Any, feat: dict) -> str:
    tier = feat.get("credit_score_tier", str(val))
    tier_desc = {
        "exceptional": "Exceptional credit score — lowest risk profile",
        "very_good":   "Very good credit score — strong approval likelihood",
        "good":        "Good credit score — meets most program guidelines",
        "fair":        "Fair credit score — may face additional conditions",
        "poor":        "Poor credit score — significant underwriting hurdle",
    }
    return tier_desc.get(tier, f"Credit tier: {tier}")


# Map feature name → (display_name, label_callable)
FEATURE_LABEL_MAP: dict[str, tuple[str, Any]] = {
    "pipeline_stage":               ("Pipeline Stage",           _stage_label),
    "days_in_current_stage":        ("Days in Current Stage",    _stage_label),
    "days_since_last_status_change":("Days Without Status Change", _days_no_change_label),
    "days_to_projected_close":      ("Days to Projected Close",  _days_to_close_label),
    "appraisal_status":             ("Appraisal Status",         _appraisal_label),
    "title_status":                 ("Title Status",             _title_label),
    "income_docs_complete":         ("Income Docs Complete",     _income_docs_label),
    "rate_lock_expiry_days":        ("Rate Lock Expiry",         _rate_lock_label),
    "condition_count":              ("Outstanding Conditions",   _condition_count_label),
    "prior_fall_out_same_stage":    ("Prior Fall-Out History",   _prior_fallout_label),
    "loan_type":                    ("Loan Type",                _loan_type_label),
    "ltv":                          ("Loan-to-Value (LTV)",      _ltv_label),
    "dti":                          ("Debt-to-Income (DTI)",     _dti_label),
    "credit_score_tier":            ("Credit Score Tier",        _credit_tier_label),
}

# ---------------------------------------------------------------------------
# Model loader (lazy singleton)
# ---------------------------------------------------------------------------
_artefact: dict | None = None


def _load_artefact() -> dict:
    global _artefact
    if _artefact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run  python model/train_pipeline_model.py  first."
            )
        with open(MODEL_PATH, "rb") as fh:
            _artefact = pickle.load(fh)
    return _artefact


def _get_explainer(model) -> shap.TreeExplainer:
    """Cache a SHAP TreeExplainer for the loaded model."""
    if not hasattr(_get_explainer, "_cache"):
        _get_explainer._cache = {}
    model_id = id(model)
    if model_id not in _get_explainer._cache:
        _get_explainer._cache[model_id] = shap.TreeExplainer(model)
    return _get_explainer._cache[model_id]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_loan(loan_features: dict) -> dict:
    """
    Compute close probability, risk level, and top-3 SHAP risk factors.

    Parameters
    ----------
    loan_features : dict
        Keys must include all model features (see FEATURE_LABEL_MAP).
        Missing numeric fields default to 0; missing categoricals default to
        the first category in the vocabulary.

    Returns
    -------
    dict with keys:
        close_probability : float  (0–1)
        risk_level        : str    ("low" | "moderate" | "high" | "critical")
        top_risk_factors  : list of dicts
            [{"factor": str, "impact": float, "plain_label": str}, ...]
        raw_shap_values   : dict   feature → shap_value (for debugging)
    """
    artefact     = _load_artefact()
    model        = artefact["model"]
    encoders     = artefact["encoders"]
    feature_names = artefact["feature_names"]

    # ---- Build a single-row DataFrame in the exact column order the model expects ----
    row: dict[str, Any] = {}

    # Ordinal features
    for feat in artefact["ordinal_features"]:
        enc = encoders[feat]
        raw = loan_features.get(feat, enc.categories_[0][0])
        row[feat] = enc.transform([[raw]])[0][0]

    # Nominal features
    for feat in artefact["nominal_features"]:
        enc = encoders[feat]
        raw = loan_features.get(feat, enc.classes_[0])
        row[feat] = enc.transform([raw])[0]

    # Numeric features
    for feat in artefact["numeric_features"]:
        row[feat] = float(loan_features.get(feat, 0))

    X = pd.DataFrame([row])[feature_names]

    # ---- 1. Predict close probability ----------------------------------------
    close_prob = float(model.predict_proba(X)[0, 1])

    # ---- 2. Risk level -------------------------------------------------------
    risk_level = "critical"
    for threshold, level in RISK_LEVELS:
        if close_prob >= threshold:
            risk_level = level
            break

    # ---- 3. SHAP values — identify features pushing probability DOWN ---------
    explainer   = _get_explainer(model)
    shap_values = explainer.shap_values(X)         # shape (1, n_features)

    if isinstance(shap_values, list):
        # Binary classification: index 1 = positive (closed) class
        sv = np.array(shap_values[1]).ravel()
    else:
        sv = np.array(shap_values).ravel()

    # Negative SHAP = hurts close probability = risk factor
    shap_dict = dict(zip(feature_names, sv))

    # Sort by SHAP value ascending (most negative = highest risk) then take top 3
    sorted_features = sorted(shap_dict.items(), key=lambda kv: kv[1])
    top_risk = [(feat, val) for feat, val in sorted_features if val < 0][:3]

    # Fallback: if fewer than 3 negative, include smallest positives
    if len(top_risk) < 3:
        remaining = [(f, v) for f, v in sorted_features if (f, v) not in top_risk]
        top_risk += remaining[: 3 - len(top_risk)]

    # ---- 4. Build plain-English labels ---------------------------------------
    top_risk_factors = []
    for feat, impact in top_risk:
        raw_val = loan_features.get(feat, None)
        display_name, label_fn = FEATURE_LABEL_MAP.get(
            feat, (feat, lambda v, _: str(v))
        )
        plain_label = label_fn(raw_val, loan_features)
        top_risk_factors.append({
            "factor":      display_name,
            "impact":      round(float(impact), 4),   # negative = risk
            "plain_label": plain_label,
        })

    return {
        "close_probability": round(close_prob, 4),
        "risk_level":        risk_level,
        "top_risk_factors":  top_risk_factors,
        "raw_shap_values":   {k: round(float(v), 4) for k, v in shap_dict.items()},
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  risk_explainer.py — demo\n")

    sample_loans = [
        {
            "name": "High-risk: stalled in underwriting",
            "loan": {
                "pipeline_stage":              "underwriting",
                "days_in_current_stage":       28,
                "days_since_last_status_change": 18,
                "days_to_projected_close":     12,
                "appraisal_status":            "disputed",
                "title_status":                "issue",
                "income_docs_complete":        0,
                "rate_lock_expiry_days":       5,
                "condition_count":             7,
                "prior_fall_out_same_stage":   1,
                "loan_type":                   "USDA",
                "ltv":                         93.5,
                "dti":                         48.2,
                "credit_score_tier":           "fair",
            },
        },
        {
            "name": "Low-risk: clear to close tomorrow",
            "loan": {
                "pipeline_stage":              "clear_to_close",
                "days_in_current_stage":       2,
                "days_since_last_status_change": 1,
                "days_to_projected_close":     3,
                "appraisal_status":            "waived",
                "title_status":                "cleared",
                "income_docs_complete":        1,
                "rate_lock_expiry_days":       30,
                "condition_count":             0,
                "prior_fall_out_same_stage":   0,
                "loan_type":                   "conventional",
                "ltv":                         72.0,
                "dti":                         33.1,
                "credit_score_tier":           "exceptional",
            },
        },
        {
            "name": "Moderate-risk: processing, some conditions",
            "loan": {
                "pipeline_stage":              "processing",
                "days_in_current_stage":       14,
                "days_since_last_status_change": 7,
                "days_to_projected_close":     35,
                "appraisal_status":            "ordered",
                "title_status":                "ordered",
                "income_docs_complete":        1,
                "rate_lock_expiry_days":       40,
                "condition_count":             3,
                "prior_fall_out_same_stage":   0,
                "loan_type":                   "FHA",
                "ltv":                         84.0,
                "dti":                         41.5,
                "credit_score_tier":           "good",
            },
        },
    ]

    for example in sample_loans:
        print(f"  {'─' * 54}")
        print(f"  {example['name']}")
        print(f"  {'─' * 54}")
        result = explain_loan(example["loan"])
        prob  = result["close_probability"]
        level = result["risk_level"].upper()
        print(f"  Close probability : {prob:.1%}  |  Risk level : {level}")
        print("  Top risk factors:")
        for i, rf in enumerate(result["top_risk_factors"], 1):
            arrow = "▼" if rf["impact"] < 0 else "▲"
            print(f"    {i}. [{rf['factor']}]  SHAP {rf['impact']:+.4f}  {arrow}")
            print(f"       {rf['plain_label']}")
        print()
