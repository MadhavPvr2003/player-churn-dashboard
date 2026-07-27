"""Data loading, churn label, and feature engineering.

Single source of truth for turning a raw player record into the feature vector
the model consumes. Training and the FastAPI service both import from here, so
the transformation can never drift between them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config


# --- Loading -----------------------------------------------------------------
def load_raw(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the player CSV and normalise column names to snake_case."""
    path = Path(csv_path) if csv_path else config.RAW_CSV
    if not path.exists():
        candidates = sorted(config.DATA_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No dataset found. Expected {config.RAW_CSV}. Generate the "
                "simulated player base with `python -m src.simulate_valorant`."
            )
        path = candidates[0]

    # keep_default_na=False so the region code "NA" (North America) is NOT
    # parsed as a null; only genuinely empty cells become NaN.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column spellings to our canonical snake_case names.

    Shared by disk loads and uploaded-CSV scoring so both go through the exact
    same normalisation (e.g. 'PlayerID', 'player id', 'player_id' -> player_id).
    """
    normalised = {}
    for col in df.columns:
        key = "".join(ch for ch in str(col).lower() if ch.isalnum())
        normalised[col] = config.RAW_COLUMNS.get(key, key)
    return df.rename(columns=normalised)


# --- Target ------------------------------------------------------------------
def derive_churn(df: pd.DataFrame) -> pd.Series:
    """Return the binary churn label (1 = lapsed / at-risk).

    Uses the precomputed `churned` column; if absent (e.g. a live scoring
    payload), derives it from match recency (> CHURN_INACTIVITY_DAYS).
    """
    if config.TARGET_COL in df.columns:
        return df[config.TARGET_COL].astype(int)
    if "days_since_last_match" in df.columns:
        return (df["days_since_last_match"] > config.CHURN_INACTIVITY_DAYS).astype(int)
    raise KeyError("No `churned` or `days_since_last_match` column to derive churn.")


# --- Feature engineering -----------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw (normalised) columns into the model's engineered feature set.

    Works on a single row or many rows, so the same path serves batch training
    and one-off API predictions. Missing raw columns get neutral defaults.
    Deliberately never touches `days_since_last_match` (the churn label source).
    """
    def num(col, default=0.0):
        return pd.to_numeric(df.get(col), errors="coerce").fillna(default)

    out = pd.DataFrame(index=df.index)

    # --- Raw pass-through numerics ---
    out["age"] = num("age", 24).clip(lower=13)
    out["account_level"] = num("account_level", 1).clip(lower=1)
    out["days_since_signup"] = num("days_since_signup", 30).clip(lower=1)
    out["hours_played"] = num("hours_played").clip(lower=0)
    out["matches_played"] = num("matches_played").clip(lower=0)
    out["sessions_per_week"] = num("sessions_per_week").clip(lower=0)
    out["avg_session_minutes"] = num("avg_session_minutes").clip(lower=0)
    out["owns_battlepass"] = num("owns_battlepass").clip(0, 1).astype(int)
    out["num_purchases"] = num("num_purchases").clip(lower=0)
    out["total_spend_usd"] = num("total_spend_usd").clip(lower=0)
    out["avg_purchase_value"] = num("avg_purchase_value").clip(lower=0)
    out["days_since_last_purchase"] = num("days_since_last_purchase", 999).clip(lower=0)

    # --- Engineered features ---
    # Weekly time invested (frequency x duration).
    out["weekly_playtime_minutes"] = out["sessions_per_week"] * out["avg_session_minutes"]
    # Monetisation intensity: $ per hour played.
    out["spend_per_hour"] = out["total_spend_usd"] / (out["hours_played"] + 1.0)
    # Purchase frequency normalised by tenure.
    out["purchases_per_month"] = out["num_purchases"] / (out["days_since_signup"] / 30.0 + 0.1)
    # Match cadence over the account's life.
    out["matches_per_active_day"] = out["matches_played"] / (out["days_since_signup"] + 1.0)
    # Session depth in match terms.
    out["hours_per_match"] = out["hours_played"] / (out["matches_played"] + 1.0)
    # Simple paying flag.
    out["is_spender"] = (out["total_spend_usd"] > 0).astype(int)

    # --- Categoricals ---
    out["region"] = df.get("region", "Unknown").astype(str).fillna("Unknown")
    out["rank"] = df.get("rank", "Unknown").astype(str).fillna("Unknown")
    out["spender_segment"] = df.get("spender_segment", "Non-spender").astype(str).fillna("Non-spender")

    return out[config.FEATURE_COLUMNS]


def build_training_frame(csv_path: Optional[Path] = None):
    """Load raw data and return (X engineered features, y churn label, raw df)."""
    raw = load_raw(csv_path)
    y = derive_churn(raw)
    X = engineer_features(raw)
    return X, y, raw
