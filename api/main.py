"""
main.py
=======
FastAPI application entry point for the Mortgage Pipeline Intelligence API.

Endpoints:
    GET  /health                      — Railway healthcheck (fast, no model needed)
    GET  /pipeline-summary            — {loans[], actions[]} for the dashboard
    GET  /loans/{loan_id}             — single loan detail + SHAP risk factors
    POST /loans/{loan_id}/explain     — re-score a loan with updated features
    PATCH /actions/{loan_id}/complete — mark an action complete

Start:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port $PORT   (production)
"""

from __future__ import annotations

import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mortgage Pipeline Intelligence API",
    description="XGBoost close-probability + SHAP risk factors + Claude AI daily actions",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

def _cors_origins() -> list[str]:
    """Build the allowed-origins list from env vars at startup."""
    origins = [
        "http://localhost:5173",   # Vite dev
        "http://localhost:4173",   # Vite preview
    ]
    # Explicit production URL, e.g. https://cu-mortgage.vercel.app
    if url := os.getenv("FRONTEND_URL", "").strip():
        origins.append(url)
    return [o for o in origins if o]   # drop empty strings


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # Allow every *.vercel.app preview deploy and any Railway/custom domain
    allow_origin_regex=r"https://.*\.(vercel\.app|railway\.app)$",
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---------------------------------------------------------------------------
# Lazy service loaders — only import heavy deps after app starts
# (keeps /health snappy even if model is not trained yet)
# ---------------------------------------------------------------------------
_model_ready = False
_explain_fn  = None
_actions_fn  = None
_ingest_fn   = None


def _load_services() -> bool:
    """Try to import ML/AI services. Returns True if model is available."""
    global _model_ready, _explain_fn, _actions_fn, _ingest_fn

    if _model_ready:
        return True

    model_path = ROOT / "model" / "pipeline_model.pkl"
    if not model_path.exists():
        log.warning("pipeline_model.pkl not found — run train_pipeline_model.py first")
        return False

    try:
        from api.risk_explainer    import explain_loan
        from api.action_generator  import generate_daily_actions
        from api.los_ingester      import ingest_records

        _explain_fn  = explain_loan
        _actions_fn  = generate_daily_actions
        _ingest_fn   = ingest_records
        _model_ready = True
        log.info("ML/AI services loaded successfully")
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load services: %s", exc)
        return False


# ---------------------------------------------------------------------------
# In-memory action completion store (replace with DB in production)
# ---------------------------------------------------------------------------
_completed_actions: set[str] = set()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RiskFactor(BaseModel):
    factor:      str
    impact:      float
    plain_label: str


class LoanAction(BaseModel):
    priority:  str
    action:    str
    owner:     str
    deadline:  str


class LoanDetail(BaseModel):
    loan_id:           str
    borrower_initials: str
    loan_amount:       float
    pipeline_stage:    str
    days_in_current_stage:          int
    days_since_last_status_change:  int
    days_to_projected_close:        int
    appraisal_status:  str
    title_status:      str
    income_docs_complete:       int
    rate_lock_expiry_days:      int
    condition_count:            int
    prior_fall_out_same_stage:  int
    loan_type:         str
    ltv:               float
    dti:               float
    credit_score_tier: str
    close_probability: float | None = None
    risk_level:        str   | None = None
    risk_factors:      list[RiskFactor] = []
    action:            LoanAction | None = None


class ActionRecord(BaseModel):
    loan_id:           str
    anon_id:           str
    priority:          str
    action:            str
    owner:             str
    deadline:          str
    close_probability: float
    risk_level:        str
    pipeline_stage:    str
    days_to_close:     int
    risk_factors:      list[str] = []
    completed:         bool = False


class PipelineSummary(BaseModel):
    loans:          list[LoanDetail]
    actions:        list[ActionRecord]
    generated_at:   str
    model_ready:    bool
    at_risk_count:  int
    total_count:    int


class CompleteRequest(BaseModel):
    completed: bool = True


class ExplainRequest(BaseModel):
    features: dict[str, Any]

# ---------------------------------------------------------------------------
# Mock data (returned when model/pkl is not yet trained)
# Matches the shape the React dashboard expects.
# ---------------------------------------------------------------------------
_MOCK_LOANS: list[dict] = [
    dict(loan_id="LN-1001", borrower_initials="A.R.", loan_amount=312000, pipeline_stage="application",    days_in_current_stage=3,  days_since_last_status_change=2,  days_to_projected_close=52, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=55, condition_count=1, prior_fall_out_same_stage=0, loan_type="conventional", ltv=76.0, dti=34.0, credit_score_tier="very_good",  close_probability=0.82, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Collect and verify 2-year employment history and W-2 documents from borrower.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-1002", borrower_initials="B.T.", loan_amount=480000, pipeline_stage="application",    days_in_current_stage=8,  days_since_last_status_change=5,  days_to_projected_close=47, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=0, rate_lock_expiry_days=50, condition_count=3, prior_fall_out_same_stage=0, loan_type="FHA",          ltv=89.0, dti=42.0, credit_score_tier="good",      close_probability=0.61, risk_level="moderate", risk_factors=[dict(factor="LTV", impact=-0.09, plain_label="LTV of 89% is high — PMI required")], action=dict(priority="high",   action="Order FHA case number and confirm MIP calculations before underwriting submission.", owner="processor", deadline="tomorrow")),
    dict(loan_id="LN-1003", borrower_initials="C.M.", loan_amount=215000, pipeline_stage="application",    days_in_current_stage=2,  days_since_last_status_change=1,  days_to_projected_close=55, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=58, condition_count=0, prior_fall_out_same_stage=0, loan_type="VA",           ltv=72.0, dti=31.0, credit_score_tier="exceptional", close_probability=0.88, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Complete VA COE verification and submit initial disclosures within 3 business days.", owner="lo", deadline="this_week")),
    dict(loan_id="LN-1004", borrower_initials="D.K.", loan_amount=560000, pipeline_stage="application",    days_in_current_stage=11, days_since_last_status_change=8,  days_to_projected_close=43, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=0, rate_lock_expiry_days=46, condition_count=5, prior_fall_out_same_stage=0, loan_type="conventional", ltv=91.0, dti=46.0, credit_score_tier="fair",       close_probability=0.44, risk_level="high",     risk_factors=[dict(factor="DTI", impact=-0.11, plain_label="DTI of 46% is elevated — above typical guideline thresholds"), dict(factor="LTV", impact=-0.09, plain_label="LTV of 91% is high — PMI required")], action=dict(priority="high", action="Schedule borrower call to review DTI reduction options before file reaches underwriting.", owner="lo", deadline="today")),
    dict(loan_id="LN-2001", borrower_initials="E.P.", loan_amount=395000, pipeline_stage="processing",     days_in_current_stage=9,  days_since_last_status_change=4,  days_to_projected_close=38, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=42, condition_count=2, prior_fall_out_same_stage=0, loan_type="conventional", ltv=81.0, dti=38.0, credit_score_tier="good",      close_probability=0.79, risk_level="moderate", risk_factors=[dict(factor="Days in Stage", impact=-0.06, plain_label="Loan has been in processing for 9 days — above average")], action=dict(priority="normal", action="Chase title company for preliminary title report — ordered 9 days ago with no update.", owner="processor", deadline="tomorrow")),
    dict(loan_id="LN-2002", borrower_initials="F.J.", loan_amount=628000, pipeline_stage="processing",     days_in_current_stage=18, days_since_last_status_change=16, days_to_projected_close=28, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=31, condition_count=4, prior_fall_out_same_stage=0, loan_type="conventional", ltv=85.0, dti=44.0, credit_score_tier="good",      close_probability=0.52, risk_level="high",     risk_factors=[dict(factor="Days Without Status Change", impact=-0.14, plain_label="No status change in 18 days — file may be stalling"), dict(factor="Outstanding Conditions", impact=-0.10, plain_label="4 outstanding conditions not yet cleared")], action=dict(priority="high", action="Escalate stalled processor file to supervisor — no movement in 18 days, 4 conditions open.", owner="management", deadline="today")),
    dict(loan_id="LN-2003", borrower_initials="G.W.", loan_amount=272000, pipeline_stage="processing",     days_in_current_stage=6,  days_since_last_status_change=3,  days_to_projected_close=41, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=45, condition_count=1, prior_fall_out_same_stage=0, loan_type="FHA",          ltv=78.0, dti=36.0, credit_score_tier="good",      close_probability=0.83, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Confirm FHA appraisal appointment is scheduled with appraiser for this week.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-2004", borrower_initials="H.N.", loan_amount=445000, pipeline_stage="processing",     days_in_current_stage=23, days_since_last_status_change=19, days_to_projected_close=22, appraisal_status="ordered",  title_status="ordered",  income_docs_complete=0, rate_lock_expiry_days=25, condition_count=7, prior_fall_out_same_stage=1, loan_type="USDA",         ltv=92.0, dti=48.0, credit_score_tier="fair",       close_probability=0.37, risk_level="critical", risk_factors=[dict(factor="Income Docs Complete", impact=-0.16, plain_label="Income documentation incomplete — required for underwriting sign-off"), dict(factor="Outstanding Conditions", impact=-0.14, plain_label="7 outstanding conditions not yet cleared")], action=dict(priority="urgent", action="Collect all 7 outstanding income documents immediately — USDA approval timeline at risk.", owner="lo", deadline="today")),
    dict(loan_id="LN-2005", borrower_initials="I.C.", loan_amount=338000, pipeline_stage="processing",     days_in_current_stage=4,  days_since_last_status_change=2,  days_to_projected_close=44, appraisal_status="received", title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=48, condition_count=0, prior_fall_out_same_stage=0, loan_type="VA",           ltv=70.0, dti=29.0, credit_score_tier="very_good",  close_probability=0.91, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Verify VA appraisal request submitted and confirm FGMC portal login for processor.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-3001", borrower_initials="J.D.", loan_amount=512000, pipeline_stage="underwriting",   days_in_current_stage=31, days_since_last_status_change=19, days_to_projected_close=9,  appraisal_status="disputed", title_status="issue",    income_docs_complete=0, rate_lock_expiry_days=4,  condition_count=9, prior_fall_out_same_stage=1, loan_type="USDA",         ltv=94.0, dti=50.0, credit_score_tier="fair",       close_probability=0.31, risk_level="critical", risk_factors=[dict(factor="Rate Lock Expiry", impact=-0.20, plain_label="Rate lock expires in 4 days — extremely tight timeline"), dict(factor="Appraisal Status", impact=-0.16, plain_label="Appraisal is currently disputed — can delay or kill the deal"), dict(factor="Outstanding Conditions", impact=-0.14, plain_label="9 outstanding conditions not yet cleared")], action=dict(priority="urgent", action="Call underwriter NOW to prioritise condition sign-off — rate lock expires in 4 days and appraisal dispute unresolved.", owner="closer", deadline="today")),
    dict(loan_id="LN-3002", borrower_initials="K.L.", loan_amount=388000, pipeline_stage="underwriting",   days_in_current_stage=14, days_since_last_status_change=7,  days_to_projected_close=22, appraisal_status="ordered",  title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=26, condition_count=3, prior_fall_out_same_stage=0, loan_type="FHA",          ltv=87.0, dti=41.0, credit_score_tier="good",      close_probability=0.68, risk_level="moderate", risk_factors=[dict(factor="Appraisal Status", impact=-0.08, plain_label="Appraisal ordered but not yet received")], action=dict(priority="normal", action="Follow up with appraiser on FHA appraisal ordered 8 days ago — confirm delivery by Friday.", owner="processor", deadline="tomorrow")),
    dict(loan_id="LN-3003", borrower_initials="L.S.", loan_amount=296000, pipeline_stage="underwriting",   days_in_current_stage=7,  days_since_last_status_change=3,  days_to_projected_close=30, appraisal_status="received", title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=34, condition_count=1, prior_fall_out_same_stage=0, loan_type="conventional", ltv=74.0, dti=33.0, credit_score_tier="very_good",  close_probability=0.85, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Clear remaining PTI condition — homeowners insurance binder not yet uploaded.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-3004", borrower_initials="M.V.", loan_amount=721000, pipeline_stage="underwriting",   days_in_current_stage=26, days_since_last_status_change=18, days_to_projected_close=11, appraisal_status="received", title_status="ordered",  income_docs_complete=1, rate_lock_expiry_days=13, condition_count=6, prior_fall_out_same_stage=0, loan_type="conventional", ltv=88.0, dti=45.0, credit_score_tier="good",      close_probability=0.43, risk_level="high",     risk_factors=[dict(factor="Days Without Status Change", impact=-0.12, plain_label="No status change in 26 days — file may be stalling"), dict(factor="Outstanding Conditions", impact=-0.11, plain_label="6 outstanding conditions not yet cleared"), dict(factor="DTI", impact=-0.09, plain_label="DTI of 45% is elevated — above typical guideline thresholds")], action=dict(priority="urgent", action="Request underwriting supervisor review — stalled 26 days with 6 conditions and lock expiring in 13 days.", owner="management", deadline="today")),
    dict(loan_id="LN-3005", borrower_initials="N.B.", loan_amount=465000, pipeline_stage="underwriting",   days_in_current_stage=11, days_since_last_status_change=5,  days_to_projected_close=18, appraisal_status="received", title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=22, condition_count=2, prior_fall_out_same_stage=1, loan_type="VA",           ltv=79.0, dti=37.0, credit_score_tier="good",      close_probability=0.72, risk_level="moderate", risk_factors=[dict(factor="Prior Fall-Out History", impact=-0.07, plain_label="Prior fall-out recorded at underwriting — elevated recurrence risk")], action=dict(priority="normal", action="Obtain updated VA appraisal NOV and confirm veteran eligibility certificate is current.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-4001", borrower_initials="O.F.", loan_amount=349000, pipeline_stage="cond_approval",  days_in_current_stage=8,  days_since_last_status_change=4,  days_to_projected_close=12, appraisal_status="received", title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=15, condition_count=3, prior_fall_out_same_stage=0, loan_type="conventional", ltv=80.0, dti=39.0, credit_score_tier="good",      close_probability=0.76, risk_level="moderate", risk_factors=[dict(factor="Rate Lock Expiry", impact=-0.10, plain_label="Rate lock expires in 15 days — tight timeline")], action=dict(priority="high", action="Collect final 3 PTD conditions and submit to underwriter today to avoid rate lock extension fee.", owner="processor", deadline="today")),
    dict(loan_id="LN-4002", borrower_initials="P.G.", loan_amount=582000, pipeline_stage="cond_approval",  days_in_current_stage=5,  days_since_last_status_change=2,  days_to_projected_close=16, appraisal_status="received", title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=20, condition_count=1, prior_fall_out_same_stage=0, loan_type="FHA",          ltv=83.0, dti=35.0, credit_score_tier="very_good",  close_probability=0.88, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Upload final homeowners insurance binder to LOS to clear last remaining PTD condition.", owner="processor", deadline="this_week")),
    dict(loan_id="LN-4003", borrower_initials="Q.H.", loan_amount=417000, pipeline_stage="cond_approval",  days_in_current_stage=15, days_since_last_status_change=11, days_to_projected_close=6,  appraisal_status="received", title_status="issue",    income_docs_complete=1, rate_lock_expiry_days=7,  condition_count=5, prior_fall_out_same_stage=0, loan_type="conventional", ltv=86.0, dti=43.0, credit_score_tier="good",      close_probability=0.58, risk_level="high",     risk_factors=[dict(factor="Rate Lock Expiry", impact=-0.18, plain_label="Rate lock expires in 7 days — tight timeline"), dict(factor="Title Status", impact=-0.12, plain_label="Title has an open issue — must be resolved before closing"), dict(factor="Outstanding Conditions", impact=-0.11, plain_label="5 outstanding conditions not yet cleared")], action=dict(priority="urgent", action="Call title company to resolve open lien on property — closing cannot proceed and lock expires in 7 days.", owner="closer", deadline="today")),
    dict(loan_id="LN-4004", borrower_initials="R.Y.", loan_amount=263000, pipeline_stage="cond_approval",  days_in_current_stage=3,  days_since_last_status_change=1,  days_to_projected_close=20, appraisal_status="waived",   title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=24, condition_count=0, prior_fall_out_same_stage=0, loan_type="VA",           ltv=71.0, dti=30.0, credit_score_tier="exceptional", close_probability=0.92, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Prepare initial CD and schedule closing disclosure review call with veteran borrower.", owner="closer", deadline="this_week")),
    dict(loan_id="LN-5001", borrower_initials="S.Z.", loan_amount=434000, pipeline_stage="clear_to_close", days_in_current_stage=1,  days_since_last_status_change=1,  days_to_projected_close=3,  appraisal_status="waived",   title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=7,  condition_count=0, prior_fall_out_same_stage=0, loan_type="conventional", ltv=75.0, dti=32.0, credit_score_tier="very_good",  close_probability=0.95, risk_level="low",      risk_factors=[dict(factor="Rate Lock Expiry", impact=-0.05, plain_label="Rate lock expires in 7 days — monitor closely")], action=dict(priority="normal", action="Confirm wire instructions with title and schedule signing for Wednesday at 2pm.", owner="closer", deadline="tomorrow")),
    dict(loan_id="LN-5002", borrower_initials="T.A.", loan_amount=371000, pipeline_stage="clear_to_close", days_in_current_stage=2,  days_since_last_status_change=1,  days_to_projected_close=2,  appraisal_status="received", title_status="cleared",  income_docs_complete=1, rate_lock_expiry_days=5,  condition_count=0, prior_fall_out_same_stage=0, loan_type="FHA",          ltv=82.0, dti=37.0, credit_score_tier="good",      close_probability=0.97, risk_level="low",      risk_factors=[], action=dict(priority="normal", action="Send final CD to borrower and confirm receipt of cashier's check for closing funds.", owner="closer", deadline="today")),
    dict(loan_id="LN-5003", borrower_initials="U.O.", loan_amount=688000, pipeline_stage="clear_to_close", days_in_current_stage=4,  days_since_last_status_change=3,  days_to_projected_close=4,  appraisal_status="received", title_status="issue",    income_docs_complete=1, rate_lock_expiry_days=5,  condition_count=1, prior_fall_out_same_stage=0, loan_type="conventional", ltv=88.0, dti=44.0, credit_score_tier="good",      close_probability=0.62, risk_level="moderate", risk_factors=[dict(factor="Rate Lock Expiry", impact=-0.15, plain_label="Rate lock expires in 5 days — extremely tight timeline"), dict(factor="Title Status", impact=-0.08, plain_label="Title has an open issue — must be resolved before closing")], action=dict(priority="urgent", action="Resolve outstanding property tax lien with title attorney — lock expires Friday and closing is blocked.", owner="closer", deadline="today")),
]

AT_RISK_THRESHOLD = float(os.getenv("AT_RISK_THRESHOLD", "0.65"))

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    log.info("Mortgage Pipeline Intelligence API starting up…")
    ready = _load_services()
    log.info("Model ready: %s", ready)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"], summary="Railway healthcheck")
def health():
    """Fast liveness probe — always returns 200 regardless of model state."""
    return {
        "status":      "ok",
        "model_ready": _model_ready,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "version":     app.version,
    }


@app.get(
    "/pipeline-summary",
    response_model=PipelineSummary,
    tags=["pipeline"],
    summary="Full pipeline snapshot — loans + daily actions",
)
def pipeline_summary():
    """
    Returns all loans with close probabilities and the prioritised action list.
    Falls back to static mock data when the model has not been trained yet.
    """
    model_ready = _load_services()

    if model_ready and _ingest_fn is not None:
        # ── Live path: score loans from CSV (if present) or use mock records ──
        csv_path = ROOT / "data" / "raw" / "pipeline_export.csv"
        if csv_path.exists():
            df = _ingest_fn(str(csv_path))
        else:
            # No real LOS export yet — score the mock records with the real model
            df = pd.DataFrame([
                {k: v for k, v in loan.items()
                 if k not in ("close_probability", "risk_level", "risk_factors", "action")}
                for loan in _MOCK_LOANS
            ])

        # Score every loan
        loans_out: list[dict] = []
        for _, row in df.iterrows():
            loan_dict = row.to_dict()
            try:
                result = _explain_fn(loan_dict)
                loan_dict["close_probability"] = result["close_probability"]
                loan_dict["risk_level"]        = result["risk_level"]
                loan_dict["risk_factors"]      = result["top_risk_factors"]
                loan_dict["action"]            = None  # actions generated separately
            except Exception as exc:  # noqa: BLE001
                log.warning("Score failed for %s: %s", loan_dict.get("loan_id"), exc)
                loan_dict.setdefault("close_probability", 0.5)
                loan_dict.setdefault("risk_level", "high")
                loan_dict.setdefault("risk_factors", [])
                loan_dict.setdefault("action", None)
            loans_out.append(loan_dict)

        # Generate actions for at-risk loans
        at_risk_df = df[
            [l["close_probability"] < AT_RISK_THRESHOLD for l in loans_out]
        ]
        actions_raw = _actions_fn(at_risk_df) if not at_risk_df.empty else []

    else:
        # ── Mock path ──────────────────────────────────────────────────────
        loans_out   = list(_MOCK_LOANS)
        actions_raw = [
            {
                **loan["action"],
                "loan_id":           loan["loan_id"],
                "anon_id":           loan["loan_id"],
                "close_probability": loan["close_probability"],
                "risk_level":        loan["risk_level"],
                "pipeline_stage":    loan["pipeline_stage"],
                "days_to_close":     loan["days_to_projected_close"],
                "risk_factors":      [f["plain_label"] for f in loan["risk_factors"]],
            }
            for loan in _MOCK_LOANS
            if loan["close_probability"] < AT_RISK_THRESHOLD
        ]
        # Sort: urgent → high → normal, then probability ascending
        _ORDER = {"urgent": 0, "high": 1, "normal": 2}
        actions_raw.sort(key=lambda a: (
            _ORDER.get(a["priority"], 9),
            a["close_probability"],
        ))

    # Annotate completed status
    actions_out = [
        {**a, "completed": a["loan_id"] in _completed_actions}
        for a in actions_raw
    ]

    at_risk = sum(1 for l in loans_out if (l.get("close_probability") or 1.0) < AT_RISK_THRESHOLD)

    return {
        "loans":         loans_out,
        "actions":       actions_out,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "model_ready":   model_ready,
        "at_risk_count": at_risk,
        "total_count":   len(loans_out),
    }


@app.get(
    "/loans/{loan_id}",
    response_model=LoanDetail,
    tags=["pipeline"],
    summary="Single loan detail with SHAP risk factors",
)
def get_loan(loan_id: str):
    """Return a single loan's detail and SHAP-scored risk factors."""
    # Find in mock data (or real data when live)
    match = next((l for l in _MOCK_LOANS if l["loan_id"] == loan_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Loan '{loan_id}' not found")

    loan_dict = dict(match)

    if _load_services() and _explain_fn is not None:
        try:
            result = _explain_fn(loan_dict)
            loan_dict["close_probability"] = result["close_probability"]
            loan_dict["risk_level"]        = result["risk_level"]
            loan_dict["risk_factors"]      = result["top_risk_factors"]
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP explain failed for %s: %s", loan_id, exc)

    return loan_dict


@app.post(
    "/loans/{loan_id}/explain",
    tags=["pipeline"],
    summary="Re-score a loan with provided feature overrides",
)
def explain_loan_endpoint(loan_id: str, body: ExplainRequest):
    """Useful for 'what-if' analysis — pass updated features and get a new score."""
    if not _load_services() or _explain_fn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready — run train_pipeline_model.py first",
        )
    try:
        result = _explain_fn(body.features)
        return {"loan_id": loan_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch(
    "/actions/{loan_id}/complete",
    tags=["actions"],
    summary="Mark a daily action as complete",
)
def mark_action_complete(loan_id: str, body: CompleteRequest):
    """Toggle completion state for a loan's daily action."""
    if body.completed:
        _completed_actions.add(loan_id)
    else:
        _completed_actions.discard(loan_id)
    return {"loan_id": loan_id, "completed": body.completed}
