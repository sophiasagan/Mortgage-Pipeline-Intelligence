# 🏦 Mortgage Pipeline Intelligence

> **XGBoost close-probability scoring · SHAP risk explanations · Claude AI daily actions · React heatmap dashboard**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=flat-square)](https://shap.readthedocs.io)
[![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-D97706?style=flat-square&logo=anthropic&logoColor=white)](https://anthropic.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev)
[![SendGrid](https://img.shields.io/badge/SendGrid-Email_digest-1A82E2?style=flat-square&logo=sendgrid&logoColor=white)](https://sendgrid.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## 📸 Dashboard Preview

> **Screenshot coming soon** — run the app locally and capture `http://localhost:5173`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🏦  Mortgage Pipeline Intelligence          ↻ Refresh   May 24, 2026       │
├──────────────┬────────────────┬──────────────────┬──────────────────────────┤
│  💰 $48.2M   │  📊 71% avg   │  ⚠️ 12 at-risk   │  🎯 8 projected closings │
│  Total Pipeline│  close prob  │  < 65% threshold │  CTC + CA ≤ 30 days      │
├──────────────┴────────────────┴──────────────────┴──────────────────────────┤
│  🔀 Stage Funnel                                                              │
│  [Application]›82%›[Processing]›79%›[Underwriting]›82%›[CA]›89%›[CTC]       │
│     34 · $9M       28 · $11M        22 · $10M    🔴18 · $8M   8 · $3M        │
│     ✅ 4d/5d        🔴 19d/14d       ✅ 18d/21d   ✅ 8d/10d    ✅ 2d/3d       │
├────────────────────────────────────────────┬─────────────────────────────────┤
│  🗺 Pipeline Heatmap (5 stage columns)     │  📋 Daily Actions               │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ...  │  🔴 URGENT  Call underwriter…   │
│  │LN-301│ │LN-201│ │LN-101│ │LN-401│      │       Closer · Today            │
│  │ J.D. │ │ H.N. │ │ M.V. │ │ Q.H. │      │  ────────────────────────────── │
│  │ 31%  │ │ 37%  │ │ 43%  │ │ 58%  │      │  🟡 HIGH   Escalate stalled…    │
│  └──────┘ └──────┘ └──────┘ └──────┘      │       Management · Today        │
│  hover → tooltip   click → sidebar ──────►  │  🔵 NORMAL  Chase title co…   │
│    ┌────────────────────────────────────┐  │       Processor · Tomorrow      │
│    │ LN-3001  ████████████░░░░░░  31%  │  │  ────────────────────────────── │
│    │ Rate lock: 4d  Conditions: 9       │  │  ✓ Mark complete  ↗ Send email  │
│    │ ▼ Rate lock expires in 4 days      │  │                                 │
│    │ ▼ Appraisal disputed               │  │                                 │
│    │ ▼ 9 outstanding conditions         │  │                                 │
│    │ [✉️ Send to Closer]  [Close]        │  │                                 │
│    └────────────────────────────────────┘  │                                 │
└────────────────────────────────────────────┴─────────────────────────────────┘
```

*Replace this block with `docs/demo.png` once captured.*

---

## ✨ Features

| Layer | What it does |
|---|---|
| **ML model** | XGBoost classifier trained on 2,000 synthetic loans predicts close probability (0–1) per loan |
| **Risk explainer** | SHAP `TreeExplainer` identifies the top-3 features dragging probability down; maps each to plain English |
| **AI actions** | Claude (`claude-sonnet-4-6`) reads stage + risk factors and generates a specific, owner-attributed action with priority and deadline |
| **Heatmap** | React grid of all loans, colour-coded green→red by probability; hover tooltip; click-to-drill sidebar |
| **Stage funnel** | Count, $ volume, avg-days-vs-benchmark, and inter-stage conversion rate for all 5 stages |
| **Action list** | Sorted urgent→high→normal; mark-complete toggle; one-click "Send to [Owner]" pre-fills email |
| **Daily digest** | SendGrid HTML email: top-10 actions, summary counts, risk factors per action |
| **MLflow** | Every training run logged: params, ROC-AUC, model artifact |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Mortgage Pipeline Intelligence                     │
└──────────────────────────────────────────────────────────────────────────┘

  ┌─── Data layer ──────────────────────────────────────────────────────┐
  │                                                                      │
  │   LOS export CSV/API                                                 │
  │         │                                                            │
  │         ▼                                                            │
  │   api/los_ingester.py ──► pandas DataFrame (normalised loan rows)   │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─── ML layer ────────────────────────────────────────────────────────┐
  │                                                                      │
  │   model/train_pipeline_model.py                                      │
  │   ├── generate_synthetic_loans()   2,000 rows w/ realistic fallout  │
  │   ├── OrdinalEncoder / LabelEncoder  on categoricals                 │
  │   ├── XGBClassifier  (scale_pos_weight, early stopping)             │
  │   ├── MLflow  log_params + log_metric(roc_auc) + log_model          │
  │   └── ► model/pipeline_model.pkl  {model, encoders, feature_names}  │
  │                                                                      │
  │   api/risk_explainer.py                                              │
  │   ├── explain_loan(loan_features) → loads pkl (lazy singleton)      │
  │   ├── model.predict_proba()       → close_probability               │
  │   ├── shap.TreeExplainer          → per-feature SHAP values         │
  │   ├── top-3 negative SHAP         → risk factors                    │
  │   └── ► {close_probability, risk_level, top_risk_factors[]}         │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─── AI action layer ─────────────────────────────────────────────────┐
  │                                                                      │
  │   api/action_generator.py                                            │
  │   ├── generate_daily_actions(pipeline_df)                           │
  │   │   ├── filter: close_probability < 0.65                          │
  │   │   ├── anonymise loan IDs  (never sent to external API)          │
  │   │   ├── Claude API  claude-sonnet-4-6                             │
  │   │   │   ├── system prompt  ──► prompt cache (ephemeral)           │
  │   │   │   └── per-loan user msg: stage, prob, 9 risk signals        │
  │   │   ├── parse JSON  {priority, action, owner, deadline}           │
  │   │   └── sort: urgent → high → normal                              │
  │   └── send_daily_digest()  ──► SendGrid HTML + plain-text email     │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─── API layer ───────────────────────────────────────────────────────┐
  │                                                                      │
  │   api/main.py  (FastAPI)                                             │
  │   ├── GET  /pipeline-summary  → {loans[], actions[]}                │
  │   ├── GET  /loans/{id}        → single loan detail + SHAP           │
  │   └── PATCH /actions/{id}/complete                                  │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
              │  REST / JSON
              ▼
  ┌─── React frontend ──────────────────────────────────────────────────┐
  │                                                                      │
  │   PipelineDashboard.jsx  (page)                                      │
  │   ├── usePipelineData()         fetch + 5-min auto-refresh          │
  │   ├── <KpiCard × 4>             $M · avg% · at-risk · closings      │
  │   ├── <StageFunnel>             count · $vol · days vs benchmark    │
  │   ├── <PipelineHeatmap>         5-column grid, colour by prob       │
  │   │   ├── <LoanCell>            bar + lock-expiry pip               │
  │   │   ├── <Tooltip>             viewport-clamped hover card         │
  │   │   └── <LoanSidebar>         SHAP bars · action · send email     │
  │   └── <ActionListCard>          sorted actions, sticky right panel  │
  │       ├── <FilterBar>           All / Urgent / High / Normal        │
  │       ├── <SummaryBar>          counts + done %                     │
  │       └── <ActionItem>          badge · complete toggle · send btn  │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  ┌─── Data flows ──────────────────────────────────────────────────────┐
  │                                                                      │
  │  Probability colour bands:                                           │
  │    ████ ≥ 80%  green   — on track                                   │
  │    ████ 60-80% yellow  — monitor                                     │
  │    ████ 40-60% orange  — at risk (Claude generates action)          │
  │    ████  < 40% red     — critical (urgent priority)                 │
  │                                                                      │
  │  At-risk threshold: 0.65  (configurable via AT_RISK_THRESHOLD env)  │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
cu_mortgage_pipeline/
│
├── model/
│   ├── train_pipeline_model.py   # Synthetic data gen · XGBoost · MLflow
│   └── pipeline_model.pkl        # Trained artefact (git-ignored)
│
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── los_ingester.py           # Parse LOS export CSV/API
│   ├── risk_explainer.py         # SHAP close probability + top risk factors
│   └── action_generator.py       # Claude daily actions + SendGrid digest
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   └── PipelineDashboard.jsx   # KPIs · funnel · layout
│       └── components/
│           ├── PipelineHeatmap.jsx     # Colour-coded loan grid + sidebar
│           └── ActionListCard.jsx      # Prioritised action list
│
├── docs/
│   └── demo.png                  # Dashboard screenshot (add after first run)
│
├── .env                          # API keys — never commit
├── .env.example                  # Template
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone & install Python dependencies

```bash
git clone https://github.com/your-org/cu_mortgage_pipeline.git
cd cu_mortgage_pipeline

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Set environment variables

```bash
cp .env.example .env
# then edit .env:
```

```ini
ANTHROPIC_API_KEY=sk-ant-...
SENDGRID_API_KEY=SG....
DIGEST_TO_EMAIL=pipeline-team@your-cu.org
DIGEST_FROM_EMAIL=noreply@your-cu.org    # must be a verified SendGrid sender
AT_RISK_THRESHOLD=0.65                   # optional, default 0.65
```

### 3 — Train the model

```bash
python model/train_pipeline_model.py
```

Expected output:

```
============================================================
  Mortgage Pipeline Intelligence — Model Training
============================================================

[1/5] Generating 2,000 synthetic loan records …
      Close rate: 68.3%  |  Fall-out rate: 31.7%

      Stage distribution:
        application        n= 402  close=55.0%  ███████████
        processing         n= 411  close=70.2%  ██████████████
        underwriting       n= 389  close=78.1%  ████████████████
        cond_approval      n= 403  close=88.1%  █████████████████▌
        clear_to_close     n= 395  close=96.0%  ███████████████████▏

[5/5] Saving model artefacts to model/pipeline_model.pkl …
      ✓ Saved  (842.3 KB)
```

### 4 — Run the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 5 — Run the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 6 — Test the action generator (standalone)

```bash
python api/action_generator.py
```

### 7 — Test the risk explainer (standalone)

```bash
python api/risk_explainer.py
```

---

## 🧠 ML Model Details

### Features (14 total)

| Feature | Type | Description |
|---|---|---|
| `pipeline_stage` | ordinal | application → clear_to_close |
| `days_in_current_stage` | numeric | Age in current stage |
| `days_since_last_status_change` | numeric | Stall indicator |
| `days_to_projected_close` | numeric | Urgency signal |
| `appraisal_status` | ordinal | ordered → received → disputed → waived |
| `title_status` | ordinal | ordered → cleared → issue |
| `income_docs_complete` | boolean | Documentation readiness |
| `rate_lock_expiry_days` | numeric | Lock pressure |
| `condition_count` | numeric | Underwriting conditions outstanding |
| `prior_fall_out_same_stage` | boolean | Historical recidivism |
| `loan_type` | nominal | conventional / FHA / VA / USDA |
| `ltv` | numeric | Loan-to-value ratio |
| `dti` | numeric | Debt-to-income ratio |
| `credit_score_tier` | ordinal | poor → fair → good → very_good → exceptional |

### Stage fall-out rates (encoded in synthetic data)

| Stage | Base fall-out | Typical close % |
|---|---|---|
| Application | 45% | 55% |
| Processing | 30% | 70% |
| Underwriting | 22% | 78% |
| Cond. Approval | 12% | 88% |
| Clear to Close | 4% | 96% |

### XGBoost hyperparameters

```python
XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight=<auto>,   # handles class imbalance
    early_stopping_rounds=30,
)
```

---

## 🤖 Claude Integration

Claude (`claude-sonnet-4-6`) is called once per at-risk loan to produce a structured action:

```json
{
  "priority": "urgent",
  "action": "Call underwriter NOW to prioritise condition sign-off — rate lock expires in 4 days and appraisal dispute unresolved.",
  "owner": "closer",
  "deadline": "today"
}
```

**Prompt caching** is enabled on the 450-token system prompt via `"cache_control": {"type": "ephemeral"}`. With 40 at-risk loans this yields ~39 cache hits, cutting prompt token costs by ~85%.

### Owner routing logic

| Owner | Triggered when |
|---|---|
| `processor` | Doc collection, appraisal/title ordering, condition clearing |
| `LO` | Client communication, rate lock extension, pre-approval |
| `closer` | Title issues, closing disclosure, wire/signing |
| `management` | Escalations, stalled files, compliance flags |

---

## 📧 Daily Digest Email

`send_daily_digest()` sends a responsive HTML + plain-text email via SendGrid with:

- **Subject**: `🔴 3 URGENT | Pipeline Action Digest — May 24`
- **Summary bar**: urgent / high / normal / total at-risk counts
- **Top 10 actions**: priority badge · action text · risk factors · owner · deadline · close %
- **At-risk threshold** watermark in footer

Configure the `DIGEST_TO_EMAIL` env var (or pass a list) for team distribution.

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `AT_RISK_THRESHOLD` | `0.65` | Close probability below which a loan is "at risk" |
| `ANTHROPIC_API_KEY` | — | Required for `action_generator.py` |
| `SENDGRID_API_KEY` | — | Required for email digest |
| `DIGEST_TO_EMAIL` | — | Recipient address(es) |
| `DIGEST_FROM_EMAIL` | — | Verified SendGrid sender |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Override in `action_generator.py` |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost 2.x, scikit-learn, SHAP, pandas, NumPy |
| Experiment tracking | MLflow |
| AI | Anthropic Python SDK, `claude-sonnet-4-6`, prompt caching |
| Email | SendGrid Python SDK |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | React 18, Vite |
| Styling | Inline styles (zero CSS framework dependency) |

---

## 🗺 Roadmap

- [ ] `api/main.py` — wire up FastAPI routes to connect ML + AI layers
- [ ] `api/los_ingester.py` — adapt to your LOS export format (Encompass / BytePro / MortgageDirector)
- [ ] Replace synthetic training data with historical closed/fallen-out loans
- [ ] Add `docs/demo.png` screenshot after first run
- [ ] Teams webhook integration for "Send to [Owner]" button
- [ ] WebSocket push for real-time probability updates
- [ ] Role-based access (processor view vs. management view)
- [ ] Automated daily digest via cron / scheduled FastAPI task

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built with ♥ for credit union mortgage teams · Powered by XGBoost + SHAP + Claude</sub>
</div>
# Mortgage-Pipeline-Intelligence
