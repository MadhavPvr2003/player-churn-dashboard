"""Central configuration: paths, schema, and the churn definition.

Keeping this in one place means EDA, feature engineering, training, serving, and
the pricing engine all agree on column names, the target, and where artifacts
live. The player data is a *simulated* Valorant cohort (see
src/simulate_valorant.py) grounded in real spend distributions and published
demographic statistics.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = DATA_DIR / "valorant_players.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Raw schema --------------------------------------------------------------
# Canonical (snake_case) names we normalise raw columns to.
RAW_COLUMNS = {
    "playerid": "player_id",
    "age": "age",
    "region": "region",
    "rank": "rank",
    "accountlevel": "account_level",
    "dayssincesignup": "days_since_signup",
    "hoursplayed": "hours_played",
    "matchesplayed": "matches_played",
    "sessionsperweek": "sessions_per_week",
    "avgsessionminutes": "avg_session_minutes",
    "dayssincelastmatch": "days_since_last_match",
    "ownsbattlepass": "owns_battlepass",
    "spendersegment": "spender_segment",
    "numpurchases": "num_purchases",
    "totalspendusd": "total_spend_usd",
    "avgpurchasevalue": "avg_purchase_value",
    "dayssincelastpurchase": "days_since_last_purchase",
    "churned": "churned",
}

# --- Target definition -------------------------------------------------------
# Churn is TEMPORAL: a player is churned if they have not played a match in
# CHURN_INACTIVITY_DAYS. It is precomputed in the `churned` column by the
# simulator. `days_since_last_match` (which defines it) is EXCLUDED from the
# model features so the model predicts imminent lapse from behaviour, not from
# the recency that defines the label.
TARGET_COL = "churned"
CHURN_INACTIVITY_DAYS = 14
LEAKAGE_EXCLUDE = ["days_since_last_match", "player_id", "churned"]

# --- Feature columns fed to the model ---------------------------------------
NUMERIC_FEATURES = [
    "age",
    "account_level",
    "days_since_signup",
    "hours_played",
    "matches_played",
    "sessions_per_week",
    "avg_session_minutes",
    "owns_battlepass",
    "num_purchases",
    "total_spend_usd",
    "avg_purchase_value",
    "days_since_last_purchase",
    # engineered (see src/features.py):
    "weekly_playtime_minutes",
    "spend_per_hour",
    "purchases_per_month",
    "matches_per_active_day",
    "hours_per_match",
    "is_spender",
]
CATEGORICAL_FEATURES = [
    "region",
    "rank",
    "spender_segment",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Risk tiers --------------------------------------------------------------
RISK_TIERS = [
    ("low", 0.0, 0.40),
    ("medium", 0.40, 0.70),
    ("high", 0.70, 1.01),
]


def risk_tier(probability: float) -> str:
    """Map a churn probability in [0, 1] to a low/medium/high tier."""
    for name, lo, hi in RISK_TIERS:
        if lo <= probability < hi:
            return name
    return "high"
