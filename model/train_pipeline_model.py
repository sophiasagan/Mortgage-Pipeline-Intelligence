"""
train_pipeline_model.py
=======================
Generates 2,000 synthetic historical loan records, trains an XGBClassifier
to predict close probability, logs the run to MLflow, and saves the fitted
model + feature metadata to model/pipeline_model.pkl.

Usage:
    python model/train_pipeline_model.py
"""

import os
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "pipeline_model.pkl"

# ---------------------------------------------------------------------------
# Categorical vocabularies (ordered where meaningful)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "application",
    "processing",
    "underwriting",
    "cond_approval",
    "clear_to_close",
]

APPRAISAL_STATUSES = ["ordered", "received", "disputed", "waived"]
TITLE_STATUSES = ["ordered", "cleared", "issue"]
LOAN_TYPES = ["conventional", "FHA", "VA", "USDA"]
CREDIT_SCORE_TIERS = ["poor", "fair", "good", "very_good", "exceptional"]

# ---------------------------------------------------------------------------
# Realistic fall-out rates by pipeline stage (lower = closer to closing)
# ---------------------------------------------------------------------------
STAGE_FALLOUT_BASE = {
    "application":     0.45,   # High attrition early
    "processing":      0.30,
    "underwriting":    0.22,
    "cond_approval":   0.12,
    "clear_to_close":  0.04,   # Very few fall out here
}

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_synthetic_loans(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Pipeline stage -------------------------------------------------------
    stage_idx = rng.integers(0, len(PIPELINE_STAGES), size=n)
    pipeline_stage = np.array(PIPELINE_STAGES)[stage_idx]

    # --- Days features --------------------------------------------------------
    # Later stages have loans that have been in the system longer
    stage_age_mean = np.array([5, 12, 18, 25, 32])[stage_idx]
    days_in_current_stage = _clamp(
        rng.normal(stage_age_mean, 6), 0, 90
    ).astype(int)

    days_since_last_status_change = _clamp(
        rng.exponential(scale=days_in_current_stage / 2 + 2), 0, 60
    ).astype(int)

    # Projected close further out for early stages
    stage_days_to_close_mean = np.array([55, 42, 30, 18, 6])[stage_idx]
    days_to_projected_close = _clamp(
        rng.normal(stage_days_to_close_mean, 10), 0, 90
    ).astype(int)

    # --- Appraisal status -----------------------------------------------------
    # Early stages more likely to be "ordered"; disputed is rare
    appraisal_probs = np.array([
        [0.55, 0.35, 0.05, 0.05],   # application
        [0.35, 0.50, 0.08, 0.07],   # processing
        [0.15, 0.55, 0.10, 0.20],   # underwriting
        [0.05, 0.50, 0.08, 0.37],   # cond_approval
        [0.02, 0.45, 0.05, 0.48],   # clear_to_close
    ])
    appraisal_idx = np.array([
        rng.choice(4, p=appraisal_probs[s]) for s in stage_idx
    ])
    appraisal_status = np.array(APPRAISAL_STATUSES)[appraisal_idx]

    # --- Title status ---------------------------------------------------------
    title_probs = np.array([
        [0.70, 0.20, 0.10],   # application
        [0.45, 0.45, 0.10],   # processing
        [0.20, 0.68, 0.12],   # underwriting
        [0.08, 0.82, 0.10],   # cond_approval
        [0.02, 0.93, 0.05],   # clear_to_close
    ])
    title_idx = np.array([
        rng.choice(3, p=title_probs[s]) for s in stage_idx
    ])
    title_status = np.array(TITLE_STATUSES)[title_idx]

    # --- Income docs complete -------------------------------------------------
    income_doc_prob = np.array([0.45, 0.65, 0.80, 0.90, 0.96])[stage_idx]
    income_docs_complete = rng.random(n) < income_doc_prob

    # --- Rate lock expiry days ------------------------------------------------
    rate_lock_expiry_days = _clamp(
        rng.normal(days_to_projected_close + 5, 8), 0, 60
    ).astype(int)

    # --- Condition count ------------------------------------------------------
    condition_base = np.array([0.5, 1.5, 3.0, 2.0, 0.5])[stage_idx]
    condition_count = _clamp(
        rng.poisson(condition_base), 0, 15
    ).astype(int)

    # --- Prior fall-out same stage --------------------------------------------
    prior_fo_prob = np.array([0.15, 0.12, 0.10, 0.08, 0.05])[stage_idx]
    prior_fall_out_same_stage = rng.random(n) < prior_fo_prob

    # --- Loan type ------------------------------------------------------------
    loan_type = rng.choice(LOAN_TYPES, size=n, p=[0.62, 0.22, 0.11, 0.05])

    # --- LTV (loan-to-value) --------------------------------------------------
    ltv = _clamp(rng.normal(78, 12), 50, 100)

    # --- DTI (debt-to-income) -------------------------------------------------
    dti = _clamp(rng.normal(38, 8), 20, 57)

    # --- Credit score tier ----------------------------------------------------
    credit_score_tier = rng.choice(
        CREDIT_SCORE_TIERS, size=n, p=[0.05, 0.15, 0.35, 0.30, 0.15]
    )
    credit_tier_idx = np.array([
        CREDIT_SCORE_TIERS.index(t) for t in credit_score_tier
    ])

    # ---------------------------------------------------------------------------
    # Target: closed (1) vs fell_out (0)
    # Combine stage base rate with feature risk adjustments
    # ---------------------------------------------------------------------------
    base_fallout = np.array([STAGE_FALLOUT_BASE[s] for s in pipeline_stage])

    # Risk adjustments (additive logit space)
    risk_adj = np.zeros(n)
    risk_adj += 0.15 * (days_in_current_stage > stage_age_mean + 10)     # stalling
    risk_adj += 0.12 * (days_since_last_status_change > 14)               # no movement
    risk_adj += 0.10 * (appraisal_status == "disputed")                   # disputed appraisal
    risk_adj += 0.08 * (title_status == "issue")                          # title problem
    risk_adj -= 0.10 * income_docs_complete.astype(float)                 # docs in = lower risk
    risk_adj += 0.12 * (rate_lock_expiry_days < 7)                        # lock expiring soon
    risk_adj += 0.04 * np.log1p(condition_count)                          # more conditions = higher risk
    risk_adj += 0.15 * prior_fall_out_same_stage.astype(float)            # history repeats
    risk_adj += 0.06 * (dti > 45)                                         # high DTI
    risk_adj += 0.06 * (ltv > 90)                                         # high LTV
    risk_adj -= 0.08 * (credit_tier_idx >= 3)                             # good credit helps
    risk_adj += 0.05 * (loan_type == "USDA")                              # USDA slower process
    risk_adj += 0.03 * (loan_type == "VA")                                # VA slightly longer

    # Convert risk to fall-out probability (sigmoid blend)
    adjusted_fallout = _clamp(base_fallout + risk_adj, 0.01, 0.95)
    fell_out = rng.random(n) < adjusted_fallout
    closed = (~fell_out).astype(int)

    df = pd.DataFrame({
        "pipeline_stage":             pipeline_stage,
        "days_in_current_stage":      days_in_current_stage,
        "days_since_last_status_change": days_since_last_status_change,
        "days_to_projected_close":    days_to_projected_close,
        "appraisal_status":           appraisal_status,
        "title_status":               title_status,
        "income_docs_complete":       income_docs_complete.astype(int),
        "rate_lock_expiry_days":      rate_lock_expiry_days,
        "condition_count":            condition_count,
        "prior_fall_out_same_stage":  prior_fall_out_same_stage.astype(int),
        "loan_type":                  loan_type,
        "ltv":                        ltv.round(2),
        "dti":                        dti.round(2),
        "credit_score_tier":          credit_score_tier,
        "closed":                     closed,
    })
    return df


# ---------------------------------------------------------------------------
# Feature engineering / encoding
# ---------------------------------------------------------------------------

# Ordinal mappings (preserve meaningful order)
ORDINAL_FEATURES = {
    "pipeline_stage":   PIPELINE_STAGES,
    "appraisal_status": APPRAISAL_STATUSES,
    "title_status":     TITLE_STATUSES,
    "credit_score_tier": CREDIT_SCORE_TIERS,
}

# Nominal → integer via LabelEncoder
NOMINAL_FEATURES = ["loan_type"]

NUMERIC_FEATURES = [
    "days_in_current_stage",
    "days_since_last_status_change",
    "days_to_projected_close",
    "income_docs_complete",
    "rate_lock_expiry_days",
    "condition_count",
    "prior_fall_out_same_stage",
    "ltv",
    "dti",
]


def build_encoders(df: pd.DataFrame) -> dict:
    """Fit all encoders on the full training dataset; return a dict for later use."""
    encoders = {}

    for feat, cats in ORDINAL_FEATURES.items():
        enc = OrdinalEncoder(categories=[cats], handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df[[feat]])
        encoders[feat] = enc

    for feat in NOMINAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[feat])
        encoders[feat] = le

    return encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Return a fully-numeric feature DataFrame ready for XGBoost."""
    out = df[NUMERIC_FEATURES].copy().astype(float)

    for feat, enc in encoders.items():
        if isinstance(enc, OrdinalEncoder):
            out[feat] = enc.transform(df[[feat]]).ravel()
        else:  # LabelEncoder
            out[feat] = enc.transform(df[feat])

    # Preserve column order: ordinal + nominal + numeric
    ordered_cols = list(ORDINAL_FEATURES) + NOMINAL_FEATURES + NUMERIC_FEATURES
    return out[ordered_cols]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_samples: int = 2_000, seed: int = 42):
    print("=" * 60)
    print("  Mortgage Pipeline Intelligence — Model Training")
    print("=" * 60)

    # 1. Generate data
    print(f"\n[1/5] Generating {n_samples:,} synthetic loan records …")
    df = generate_synthetic_loans(n=n_samples, seed=seed)
    close_rate = df["closed"].mean()
    print(f"      Close rate: {close_rate:.1%}  |  Fall-out rate: {1 - close_rate:.1%}")
    print("\n      Stage distribution:")
    stage_stats = df.groupby("pipeline_stage")["closed"].agg(["count", "mean"])
    stage_stats.columns = ["count", "close_rate"]
    stage_stats = stage_stats.reindex(PIPELINE_STAGES)
    for stage, row in stage_stats.iterrows():
        bar = "█" * int(row["close_rate"] * 20)
        print(f"        {stage:<18} n={int(row['count']):>4}  close={row['close_rate']:.1%}  {bar}")

    # 2. Encode
    print("\n[2/5] Encoding features …")
    encoders = build_encoders(df)
    X = encode_features(df, encoders)
    y = df["closed"]
    feature_names = list(X.columns)
    print(f"      {len(feature_names)} features: {feature_names}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    print(f"\n[3/5] Train/test split: {len(X_train):,} train / {len(X_test):,} test")

    # 4. Train with MLflow
    print("\n[4/5] Training XGBClassifier …")

    mlflow.set_experiment("mortgage_pipeline_intelligence")

    xgb_params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="xgb_pipeline_v1") as run:
        mlflow.log_params(xgb_params)
        mlflow.log_param("n_training_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("feature_names", feature_names)

        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Metrics
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)
        auc     = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("test_roc_auc", auc)
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")

        print(f"\n      ROC-AUC: {auc:.4f}")
        print("\n      Classification report:")
        report = classification_report(y_test, y_pred, target_names=["fell_out", "closed"])
        for line in report.splitlines():
            print(f"        {line}")

        print(f"\n      MLflow run ID : {run.info.run_id}")

    # 5. Persist model + encoders + metadata
    print(f"\n[5/5] Saving model artefacts to {MODEL_PATH} …")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artefact = {
        "model":         model,
        "encoders":      encoders,
        "feature_names": feature_names,
        "ordinal_features":  list(ORDINAL_FEATURES.keys()),
        "nominal_features":  NOMINAL_FEATURES,
        "numeric_features":  NUMERIC_FEATURES,
        "pipeline_stages":   PIPELINE_STAGES,
        "at_risk_threshold": 0.65,
        "train_roc_auc":     auc,
    }
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artefact, fh, protocol=5)

    print(f"      ✓ Saved  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print("\n  Training complete.\n")
    return artefact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
