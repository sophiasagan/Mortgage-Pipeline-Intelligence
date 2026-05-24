"""
action_generator.py
===================
Generates daily prioritised action lists for at-risk mortgage loans using
Claude (Anthropic API) and emails a digest via SendGrid.

Public API
----------
    generate_daily_actions(pipeline_df: pd.DataFrame) -> list[dict]

Each returned dict:
    {
        loan_id:        str   — anonymised loan reference
        priority:       str   — "urgent" | "high" | "normal"
        action:         str   — specific 1-sentence action
        owner:          str   — "processor" | "LO" | "closer" | "management"
        deadline:       str   — "today" | "tomorrow" | "this_week"
        close_probability: float
        risk_level:     str
        pipeline_stage: str
        days_to_close:  int
        risk_factors:   list[str]  — plain-English labels from SHAP
    }

Usage (standalone):
    python api/action_generator.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import textwrap
import time
from datetime import date, timedelta
from typing import Any

import anthropic
import pandas as pd

# Local import — risk_explainer must be importable from the same package dir
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from risk_explainer import explain_loan, RISK_LEVELS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AT_RISK_THRESHOLD   = float(os.getenv("AT_RISK_THRESHOLD", "0.65"))
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SENDGRID_API_KEY    = os.getenv("SENDGRID_API_KEY", "")
DIGEST_FROM_EMAIL   = os.getenv("DIGEST_FROM_EMAIL", "pipeline@cu-mortgage.internal")
DIGEST_TO_EMAIL     = os.getenv("DIGEST_TO_EMAIL", "team@cu-mortgage.internal")
CLAUDE_MODEL        = "claude-sonnet-4-6"          # fast + smart for structured tasks
MAX_RETRIES         = 3
RETRY_DELAY_S       = 2.0
TOP_N_IN_DIGEST     = 10

PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2}

# ---------------------------------------------------------------------------
# Anonymise loan IDs — never send real member IDs to an external API
# ---------------------------------------------------------------------------
_loan_id_map: dict[str, str] = {}
_loan_counter = 0


def _anonymise(loan_id: str) -> str:
    """Return a stable fake reference (e.g. 'LOAN-0042') for a real loan ID."""
    global _loan_counter
    if loan_id not in _loan_id_map:
        _loan_counter += 1
        _loan_id_map[loan_id] = f"LOAN-{_loan_counter:04d}"
    return _loan_id_map[loan_id]


def _deanonymise(fake_id: str) -> str | None:
    """Reverse lookup: fake ID → real loan ID (used internally after Claude call)."""
    reverse = {v: k for k, v in _loan_id_map.items()}
    return reverse.get(fake_id)


# ---------------------------------------------------------------------------
# Claude prompt construction
# ---------------------------------------------------------------------------

# System prompt is static → eligible for Anthropic prompt caching.
_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior mortgage operations expert helping a credit union's pipeline
    team close at-risk loans before they fall out of the pipeline.

    When given a loan summary you MUST return a single valid JSON object — no
    markdown, no prose, no extra keys — matching this exact schema:

    {
      "priority": "<urgent|high|normal>",
      "action":   "<one specific sentence starting with a verb>",
      "owner":    "<processor|LO|closer|management>",
      "deadline": "<today|tomorrow|this_week>"
    }

    Priority rules:
    • urgent  — close_probability < 0.40, OR rate lock expires ≤ 5 days,
                OR title/appraisal issue blocking clear-to-close
    • high    — close_probability 0.40–0.55, OR ≥ 4 uncleared conditions,
                OR days_to_close ≤ 7
    • normal  — close_probability 0.55–0.64 with manageable risk factors

    Owner rules:
    • processor   — document collection, ordering appraisal/title, condition clearing
    • LO          — client communication, rate lock extension, pre-approval updates
    • closer      — title issues, closing disclosure, wire/signing coordination
    • management  — escalations, critical pipeline risk, compliance flags

    Deadline rules:
    • today       — urgent situations or same-day rate lock / closing deadlines
    • tomorrow    — high priority with 1–3 day window
    • this_week   — normal priority with 4–7 day window

    Action must be specific — name the exact condition, document, party, or
    system involved. Never write generic advice like "follow up on the loan".
""").strip()


def _build_user_message(
    anon_id:       str,
    stage:         str,
    close_prob:    float,
    risk_level:    str,
    days_to_close: int,
    risk_factors:  list[str],
    loan_features: dict,
) -> str:
    """Compose the per-loan user message sent to Claude."""
    factors_text = "\n".join(f"  • {f}" for f in risk_factors) or "  • (none identified)"

    # Pull a few key numeric signals for the prompt — no PII
    rate_lock = loan_features.get("rate_lock_expiry_days", "?")
    conditions = loan_features.get("condition_count", "?")
    loan_type  = loan_features.get("loan_type", "?")
    appraisal  = loan_features.get("appraisal_status", "?")
    title      = loan_features.get("title_status", "?")
    dti        = loan_features.get("dti", "?")
    ltv        = loan_features.get("ltv", "?")
    income_ok  = bool(loan_features.get("income_docs_complete", False))
    prior_fo   = bool(loan_features.get("prior_fall_out_same_stage", False))

    return textwrap.dedent(f"""
        Loan reference   : {anon_id}
        Pipeline stage   : {stage}
        Close probability: {close_prob:.1%}  ({risk_level} risk)
        Days to close    : {days_to_close}
        Loan type        : {loan_type}
        LTV / DTI        : {ltv}% / {dti}%
        Rate lock expiry : {rate_lock} days
        Conditions open  : {conditions}
        Appraisal status : {appraisal}
        Title status     : {title}
        Income docs done : {"Yes" if income_ok else "No"}
        Prior fall-out   : {"Yes — same stage" if prior_fo else "No"}

        Top SHAP risk factors:
        {factors_text}

        Generate the JSON action object.
    """).strip()


# ---------------------------------------------------------------------------
# Claude API call (with retry + prompt caching)
# ---------------------------------------------------------------------------

def _call_claude(user_message: str, client: anthropic.Anthropic) -> dict:
    """
    Call Claude with prompt caching on the static system prompt.
    Returns parsed JSON dict or raises on repeated failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                system=[
                    {
                        "type":       "text",
                        "text":       _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},   # prompt cache
                    }
                ],
                messages=[
                    {"role": "user", "content": user_message}
                ],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            # Validate required keys
            for key in ("priority", "action", "owner", "deadline"):
                if key not in parsed:
                    raise ValueError(f"Missing key '{key}' in Claude response")

            # Normalise values
            parsed["priority"] = parsed["priority"].lower().strip()
            parsed["owner"]    = parsed["owner"].lower().strip()
            parsed["deadline"] = parsed["deadline"].lower().strip()

            if parsed["priority"] not in PRIORITY_ORDER:
                parsed["priority"] = "normal"
            if parsed["owner"] not in ("processor", "lo", "closer", "management"):
                parsed["owner"] = "processor"
            if parsed["deadline"] not in ("today", "tomorrow", "this_week"):
                parsed["deadline"] = "this_week"

            return parsed

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
            log.warning("Claude call attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)

    # Final fallback — return a safe default so the pipeline doesn't crash
    log.error("All Claude retries exhausted; using fallback action for this loan.")
    return {
        "priority": "high",
        "action":   "Review loan file manually — automated action generation failed.",
        "owner":    "processor",
        "deadline": "today",
    }


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def generate_daily_actions(pipeline_df: pd.DataFrame) -> list[dict]:
    """
    For every loan in pipeline_df with close_probability < AT_RISK_THRESHOLD,
    generate a prioritised daily action using Claude.

    Required columns in pipeline_df (matches los_ingester output):
        loan_id, pipeline_stage, days_in_current_stage,
        days_since_last_status_change, days_to_projected_close,
        appraisal_status, title_status, income_docs_complete,
        rate_lock_expiry_days, condition_count, prior_fall_out_same_stage,
        loan_type, ltv, dti, credit_score_tier

    Optional pre-computed columns (skips re-scoring if present):
        close_probability, risk_level

    Returns
    -------
    list[dict] sorted urgent → high → normal, then by close_probability asc.
    """
    if pipeline_df.empty:
        log.info("Pipeline DataFrame is empty — nothing to process.")
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    actions: list[dict] = []
    at_risk_count = 0

    for _, row in pipeline_df.iterrows():
        loan_id = str(row.get("loan_id", f"unknown-{_}"))

        # Build feature dict for explain_loan (handles encoding internally)
        loan_features = {
            "pipeline_stage":                row.get("pipeline_stage", "processing"),
            "days_in_current_stage":         float(row.get("days_in_current_stage", 0)),
            "days_since_last_status_change": float(row.get("days_since_last_status_change", 0)),
            "days_to_projected_close":       float(row.get("days_to_projected_close", 30)),
            "appraisal_status":              row.get("appraisal_status", "ordered"),
            "title_status":                  row.get("title_status", "ordered"),
            "income_docs_complete":          int(row.get("income_docs_complete", 0)),
            "rate_lock_expiry_days":         float(row.get("rate_lock_expiry_days", 30)),
            "condition_count":               int(row.get("condition_count", 0)),
            "prior_fall_out_same_stage":     int(row.get("prior_fall_out_same_stage", 0)),
            "loan_type":                     row.get("loan_type", "conventional"),
            "ltv":                           float(row.get("ltv", 80)),
            "dti":                           float(row.get("dti", 38)),
            "credit_score_tier":             row.get("credit_score_tier", "good"),
        }

        # Use pre-computed probability if available (avoids redundant model calls)
        if "close_probability" in row and pd.notna(row["close_probability"]):
            close_prob = float(row["close_probability"])
            risk_level = row.get("risk_level", "high")
            risk_factors = list(row.get("risk_factors", []))
        else:
            explanation = explain_loan(loan_features)
            close_prob   = explanation["close_probability"]
            risk_level   = explanation["risk_level"]
            risk_factors = [rf["plain_label"] for rf in explanation["top_risk_factors"]]

        # Skip loans that are not at risk
        if close_prob >= AT_RISK_THRESHOLD:
            continue

        at_risk_count += 1
        anon_id = _anonymise(loan_id)
        stage   = loan_features["pipeline_stage"]
        days_to_close = int(loan_features["days_to_projected_close"])

        log.info(
            "  Processing %s (%s)  prob=%.1f%%  risk=%s  days_to_close=%d",
            anon_id, stage, close_prob * 100, risk_level, days_to_close,
        )

        user_msg = _build_user_message(
            anon_id=anon_id,
            stage=stage,
            close_prob=close_prob,
            risk_level=risk_level,
            days_to_close=days_to_close,
            risk_factors=risk_factors,
            loan_features=loan_features,
        )

        claude_action = _call_claude(user_msg, client)

        actions.append({
            "loan_id":           loan_id,       # real ID stays internal
            "anon_id":           anon_id,        # used in email digest
            "priority":          claude_action["priority"],
            "action":            claude_action["action"],
            "owner":             claude_action["owner"],
            "deadline":          claude_action["deadline"],
            "close_probability": close_prob,
            "risk_level":        risk_level,
            "pipeline_stage":    stage,
            "days_to_close":     days_to_close,
            "risk_factors":      risk_factors,
        })

    log.info(
        "Processed %d at-risk loans (of %d total).  %d actions generated.",
        at_risk_count, len(pipeline_df), len(actions),
    )

    # Sort: urgent → high → normal, then by close_probability ascending (worst first)
    actions.sort(key=lambda a: (
        PRIORITY_ORDER.get(a["priority"], 99),
        a["close_probability"],
    ))

    # Send digest (fire-and-forget; errors are logged, not re-raised)
    try:
        send_daily_digest(actions)
    except Exception as exc:  # noqa: BLE001
        log.error("Digest email failed (non-fatal): %s", exc)

    return actions


# ---------------------------------------------------------------------------
# SendGrid digest
# ---------------------------------------------------------------------------

_PRIORITY_COLOR = {
    "urgent": "#C0392B",
    "high":   "#E67E22",
    "normal": "#2980B9",
}

_DEADLINE_BADGE = {
    "today":     "🔴 Today",
    "tomorrow":  "🟡 Tomorrow",
    "this_week": "🟢 This Week",
}

_OWNER_ICON = {
    "processor":  "📋",
    "lo":         "📞",
    "closer":     "🖊️",
    "management": "🚨",
}


def _build_html_digest(actions: list[dict], today: date) -> str:
    top = actions[:TOP_N_IN_DIGEST]
    date_str = today.strftime("%A, %B %-d, %Y")

    # Summary counts
    urgent_n = sum(1 for a in actions if a["priority"] == "urgent")
    high_n   = sum(1 for a in actions if a["priority"] == "high")
    normal_n = sum(1 for a in actions if a["priority"] == "normal")

    rows_html = ""
    for i, a in enumerate(top, 1):
        color     = _PRIORITY_COLOR.get(a["priority"], "#555")
        deadline  = _DEADLINE_BADGE.get(a["deadline"], a["deadline"])
        icon      = _OWNER_ICON.get(a["owner"], "👤")
        factors   = "<br>".join(
            f"&nbsp;&nbsp;• {f}" for f in (a.get("risk_factors") or [])
        ) or "&nbsp;&nbsp;• (none)"
        prob_pct  = f"{a['close_probability']:.0%}"

        rows_html += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:14px 10px; text-align:center; font-size:18px; color:#999;">{i}</td>
          <td style="padding:14px 6px;">
            <span style="
              display:inline-block; padding:3px 9px; border-radius:12px;
              background:{color}; color:#fff; font-size:11px; font-weight:700;
              text-transform:uppercase; letter-spacing:.5px;
            ">{a['priority']}</span>
          </td>
          <td style="padding:14px 8px; font-size:13px; color:#333; max-width:340px;">
            <strong>{a['action']}</strong>
            <div style="margin-top:6px; font-size:11px; color:#777; line-height:1.5;">
              {factors}
            </div>
          </td>
          <td style="padding:14px 8px; font-size:12px; color:#555; white-space:nowrap;">
            {icon} {a['owner'].upper()}
          </td>
          <td style="padding:14px 8px; font-size:12px; white-space:nowrap;">{deadline}</td>
          <td style="padding:14px 8px; font-size:12px; color:#555; white-space:nowrap;">
            {a['pipeline_stage'].replace('_',' ').title()}
          </td>
          <td style="padding:14px 8px; font-size:12px; text-align:center;">
            <span style="
              font-weight:700;
              color:{'#C0392B' if a['close_probability'] < 0.40 else '#E67E22' if a['close_probability'] < 0.55 else '#888'};
            ">{prob_pct}</span>
          </td>
        </tr>
        """

    total_shown = len(top)
    remaining   = max(0, len(actions) - TOP_N_IN_DIGEST)
    footer_note = (
        f"<p style='color:#999;font-size:12px;text-align:center;'>"
        f"Showing top {total_shown} of {len(actions)} at-risk actions. "
        + (f"{remaining} additional action(s) not shown." if remaining else "")
        + "</p>"
    )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Pipeline Action Digest</title></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
             background:#f5f5f5; margin:0; padding:24px;">
  <div style="max-width:860px; margin:0 auto; background:#fff;
              border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.12); overflow:hidden;">

    <!-- Header -->
    <div style="background:#1A3C5E; padding:24px 28px; color:#fff;">
      <h1 style="margin:0 0 4px; font-size:22px; font-weight:700;">
        🏦 Mortgage Pipeline — Daily Action Digest
      </h1>
      <p style="margin:0; font-size:14px; opacity:.8;">{date_str}</p>
    </div>

    <!-- Summary bar -->
    <div style="display:flex; gap:0; border-bottom:1px solid #eee;">
      <div style="flex:1; padding:16px 20px; text-align:center; border-right:1px solid #eee;">
        <div style="font-size:28px; font-weight:700; color:#C0392B;">{urgent_n}</div>
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:.5px;">Urgent</div>
      </div>
      <div style="flex:1; padding:16px 20px; text-align:center; border-right:1px solid #eee;">
        <div style="font-size:28px; font-weight:700; color:#E67E22;">{high_n}</div>
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:.5px;">High</div>
      </div>
      <div style="flex:1; padding:16px 20px; text-align:center; border-right:1px solid #eee;">
        <div style="font-size:28px; font-weight:700; color:#2980B9;">{normal_n}</div>
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:.5px;">Normal</div>
      </div>
      <div style="flex:1; padding:16px 20px; text-align:center;">
        <div style="font-size:28px; font-weight:700; color:#555;">{len(actions)}</div>
        <div style="font-size:12px; color:#888; text-transform:uppercase; letter-spacing:.5px;">Total At-Risk</div>
      </div>
    </div>

    <!-- Action table -->
    <div style="overflow-x:auto;">
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr style="background:#f9f9f9; border-bottom:2px solid #e0e0e0;">
            <th style="padding:10px 10px; text-align:center; color:#888; font-weight:600; font-size:11px;">#</th>
            <th style="padding:10px 6px; text-align:left; color:#888; font-weight:600; font-size:11px;">PRI</th>
            <th style="padding:10px 8px; text-align:left; color:#888; font-weight:600; font-size:11px;">ACTION &amp; RISK FACTORS</th>
            <th style="padding:10px 8px; text-align:left; color:#888; font-weight:600; font-size:11px;">OWNER</th>
            <th style="padding:10px 8px; text-align:left; color:#888; font-weight:600; font-size:11px;">DEADLINE</th>
            <th style="padding:10px 8px; text-align:left; color:#888; font-weight:600; font-size:11px;">STAGE</th>
            <th style="padding:10px 8px; text-align:center; color:#888; font-weight:600; font-size:11px;">CLOSE %</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    {footer_note}

    <!-- Footer -->
    <div style="padding:16px 28px; border-top:1px solid #eee; font-size:11px; color:#aaa; text-align:center;">
      Generated by Mortgage Pipeline Intelligence &nbsp;|&nbsp;
      At-risk threshold: &lt; {AT_RISK_THRESHOLD:.0%} close probability &nbsp;|&nbsp;
      Do not reply to this automated message.
    </div>
  </div>
</body>
</html>"""


def _build_text_digest(actions: list[dict], today: date) -> str:
    """Plain-text fallback for email clients that don't render HTML."""
    lines = [
        f"MORTGAGE PIPELINE — DAILY ACTION DIGEST",
        f"{today.strftime('%A, %B %d, %Y')}",
        "=" * 60,
        "",
    ]
    urgent_n = sum(1 for a in actions if a["priority"] == "urgent")
    high_n   = sum(1 for a in actions if a["priority"] == "high")
    normal_n = sum(1 for a in actions if a["priority"] == "normal")
    lines += [
        f"At-Risk Summary:  {urgent_n} URGENT  /  {high_n} HIGH  /  {normal_n} NORMAL  ({len(actions)} total)",
        "",
        f"Top {min(TOP_N_IN_DIGEST, len(actions))} Actions:",
        "-" * 60,
    ]
    for i, a in enumerate(actions[:TOP_N_IN_DIGEST], 1):
        deadline = {"today": "TODAY", "tomorrow": "TOMORROW", "this_week": "THIS WEEK"}.get(
            a["deadline"], a["deadline"].upper()
        )
        lines += [
            f"{i:>2}. [{a['priority'].upper()}] {a['action']}",
            f"     Owner: {a['owner'].upper()}  |  Deadline: {deadline}  |  "
            f"Stage: {a['pipeline_stage']}  |  Close: {a['close_probability']:.0%}",
            "",
        ]
    lines += ["—", "Mortgage Pipeline Intelligence — automated digest"]
    return "\n".join(lines)


def send_daily_digest(
    actions: list[dict],
    to_email:   str | None = None,
    from_email: str | None = None,
) -> bool:
    """
    Send the top-10 action digest via SendGrid.

    Returns True on success, False on failure (errors are logged, not raised).
    Requires:
        SENDGRID_API_KEY env var
        DIGEST_TO_EMAIL  env var  (or pass to_email)
        DIGEST_FROM_EMAIL env var (or pass from_email)
    """
    try:
        import sendgrid                             # pip install sendgrid
        from sendgrid.helpers.mail import (
            Mail, To, From, Subject, HtmlContent, Content, MimeType,
        )
    except ImportError:
        log.error(
            "sendgrid package not installed. Run: pip install sendgrid\n"
            "Digest email skipped."
        )
        return False

    api_key = SENDGRID_API_KEY
    if not api_key:
        log.warning("SENDGRID_API_KEY not set — digest email skipped.")
        return False

    to_addr   = to_email   or DIGEST_TO_EMAIL
    from_addr = from_email or DIGEST_FROM_EMAIL
    today     = date.today()

    if not actions:
        log.info("No at-risk actions — digest email not sent.")
        return False

    urgent_n = sum(1 for a in actions if a["priority"] == "urgent")
    subject_tag = f"🔴 {urgent_n} URGENT" if urgent_n else f"⚠️ {len(actions)} At-Risk"
    subject = (
        f"{subject_tag} | Pipeline Action Digest — "
        f"{today.strftime('%b %-d')}"
    )

    html_body = _build_html_digest(actions, today)
    text_body = _build_text_digest(actions, today)

    message = Mail(
        from_email=From(from_addr, "Pipeline Intelligence"),
        to_emails=To(to_addr),
        subject=Subject(subject),
    )
    message.content = [
        Content(MimeType.text, text_body),
        Content(MimeType.html, html_body),
    ]

    try:
        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        response = sg.client.mail.send.post(request_body=message.get())
        status = response.status_code
        if 200 <= status < 300:
            log.info("Digest email sent → %s  (HTTP %d)", to_addr, status)
            return True
        else:
            log.error("SendGrid returned HTTP %d", status)
            return False
    except Exception as exc:  # noqa: BLE001
        log.error("SendGrid send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI demo — runs against a small synthetic pipeline snapshot
# ---------------------------------------------------------------------------

def _make_demo_pipeline() -> pd.DataFrame:
    """Build a tiny representative pipeline for smoke-testing."""
    return pd.DataFrame([
        # --- URGENT: critical probability, expiring lock, disputed appraisal ---
        dict(
            loan_id="LN-20001", pipeline_stage="underwriting",
            days_in_current_stage=31, days_since_last_status_change=19,
            days_to_projected_close=9,  appraisal_status="disputed",
            title_status="issue",       income_docs_complete=0,
            rate_lock_expiry_days=4,    condition_count=8,
            prior_fall_out_same_stage=1, loan_type="USDA",
            ltv=94.0, dti=49.5, credit_score_tier="fair",
        ),
        # --- HIGH: rate lock tight, conditions stacking ---
        dict(
            loan_id="LN-20002", pipeline_stage="cond_approval",
            days_in_current_stage=22, days_since_last_status_change=12,
            days_to_projected_close=6,  appraisal_status="received",
            title_status="ordered",     income_docs_complete=1,
            rate_lock_expiry_days=8,    condition_count=5,
            prior_fall_out_same_stage=0, loan_type="FHA",
            ltv=88.5, dti=44.0, credit_score_tier="good",
        ),
        # --- HIGH: stalled in processing, income docs missing ---
        dict(
            loan_id="LN-20003", pipeline_stage="processing",
            days_in_current_stage=24, days_since_last_status_change=16,
            days_to_projected_close=28, appraisal_status="ordered",
            title_status="ordered",     income_docs_complete=0,
            rate_lock_expiry_days=33,   condition_count=4,
            prior_fall_out_same_stage=1, loan_type="VA",
            ltv=82.0, dti=42.5, credit_score_tier="fair",
        ),
        # --- NORMAL: cond_approval, manageable ---
        dict(
            loan_id="LN-20004", pipeline_stage="cond_approval",
            days_in_current_stage=14, days_since_last_status_change=6,
            days_to_projected_close=14, appraisal_status="received",
            title_status="cleared",     income_docs_complete=1,
            rate_lock_expiry_days=20,   condition_count=3,
            prior_fall_out_same_stage=0, loan_type="conventional",
            ltv=78.0, dti=39.0, credit_score_tier="very_good",
        ),
        # --- Safe (should be filtered out — not at risk) ---
        dict(
            loan_id="LN-20005", pipeline_stage="clear_to_close",
            days_in_current_stage=2,  days_since_last_status_change=1,
            days_to_projected_close=3, appraisal_status="waived",
            title_status="cleared",    income_docs_complete=1,
            rate_lock_expiry_days=30,  condition_count=0,
            prior_fall_out_same_stage=0, loan_type="conventional",
            ltv=72.0, dti=32.0, credit_score_tier="exceptional",
        ),
    ])


if __name__ == "__main__":
    import os

    # Allow override via env; warn if key is missing
    if not ANTHROPIC_API_KEY:
        print("\n  ⚠  ANTHROPIC_API_KEY not set in environment.")
        print("     Set it and re-run, or the script will fail on Claude calls.\n")

    print("=" * 62)
    print("  action_generator.py — CLI demo")
    print("=" * 62)

    demo_df = _make_demo_pipeline()
    print(f"\n  Pipeline snapshot: {len(demo_df)} loans")
    print(f"  At-risk threshold: < {AT_RISK_THRESHOLD:.0%}\n")

    actions = generate_daily_actions(demo_df)

    if not actions:
        print("\n  No at-risk loans found in demo snapshot.")
    else:
        print(f"\n  {'─' * 60}")
        print(f"  {len(actions)} action(s) generated — sorted by priority\n")
        for i, a in enumerate(actions, 1):
            pri_icons = {"urgent": "🔴", "high": "🟡", "normal": "🟢"}
            icon      = pri_icons.get(a["priority"], "⚪")
            deadline  = {"today": "TODAY", "tomorrow": "TOMORROW",
                         "this_week": "THIS WEEK"}.get(a["deadline"], a["deadline"])
            print(
                f"  {i:>2}. {icon} [{a['priority'].upper():6s}]  "
                f"{a['anon_id']}  ({a['pipeline_stage']})  "
                f"prob={a['close_probability']:.0%}"
            )
            print(f"       → {a['action']}")
            print(f"         Owner: {a['owner'].upper()}   Deadline: {deadline}")
            if a.get("risk_factors"):
                for rf in a["risk_factors"]:
                    print(f"         • {rf}")
            print()

    print("  Done.\n")
