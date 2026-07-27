"""Night Market pricing engine: turn a churn score into a retention offer.

Valorant already ships a personalised-discount feature — the **Night Market** —
that shows each player randomised skin discounts. This module makes that smart:
for each player it picks the offer (a priced in-game purchase / discount / pass)
that maximises **expected retained revenue**, weighing three signals the user
asked for — churn risk, purchase averaging (real spend), and age.

── HONEST SCOPE ─────────────────────────────────────────────────────────────
Offer acceptance rates and retention lifts are business **assumptions** (there
is no offer/experiment log to learn them from) — they live in `OFFERS` and are
what you'd replace with A/B-test or uplift-model results. What IS grounded in
data: the player's realised monthly spend (from the simulated real-$ history)
and the age→spend curve in src/calibration.py. The framework — value × churn ×
expected-lift optimisation — is the transferable part.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src import calibration

HORIZON_MONTHS = 6                 # value horizon for a retained player
NONSPENDER_CONVERSION_VALUE = 0.15  # expected fraction of age-spend a F2P yields

# Valorant-authentic retention offers. `margin` = expected net $ per acceptance;
# `retention_lift` = relative churn reduction for an accepter; `kind` drives how
# acceptance is modulated. Prices reflect real Valorant economy (skins ~$10–25,
# battlepass ~$10, premium bundles ~$60–100).
OFFERS: List[Dict] = [
    {"id": "none", "name": "No offer (organic)", "kind": "none",
     "margin": 0.0, "retention_lift": 0.0,
     "base_accept_spender": 0.0, "base_accept_nonspender": 0.0, "price_point": 0},

    {"id": "nm_15", "name": "Night Market — 15% off a skin", "kind": "discount",
     "margin": 14.0, "retention_lift": 0.12,
     "base_accept_spender": 0.30, "base_accept_nonspender": 0.10, "price_point": 18},

    {"id": "nm_40", "name": "Night Market — 40% off (win-back)", "kind": "discount",
     "margin": 8.0, "retention_lift": 0.28,
     "base_accept_spender": 0.35, "base_accept_nonspender": 0.22, "price_point": 12},

    {"id": "battlepass", "name": "Discounted Battlepass ($6.49)", "kind": "engagement",
     "margin": 6.0, "retention_lift": 0.35,
     "base_accept_spender": 0.28, "base_accept_nonspender": 0.14, "price_point": 6},

    {"id": "vp_bonus", "name": "Bonus VP on next top-up", "kind": "spender",
     "margin": 10.0, "retention_lift": 0.10,
     "base_accept_spender": 0.25, "base_accept_nonspender": 0.03, "price_point": 10},

    {"id": "bundle", "name": "Premium Skin Bundle ($20 off)", "kind": "premium",
     "margin": 30.0, "retention_lift": 0.15,
     "base_accept_spender": 0.12, "base_accept_nonspender": 0.02, "price_point": 80},
]


def estimate_monthly_value(row: pd.Series) -> float:
    """Player's monthly value from REAL spend history; age-based for F2P."""
    spend = float(row.get("total_spend_usd", 0) or 0)
    months = max(float(row.get("days_since_signup", 30) or 30) / 30.0, 1.0)
    if spend > 0:
        return spend / months                       # realised monthly spend
    # Non-spender: expected value if converted, discounted by conversion odds.
    age = int(row.get("age", 24) or 24)
    return calibration.expected_monthly_spend(age) * NONSPENDER_CONVERSION_VALUE


def _age_factor(age: int) -> float:
    """Younger players are more discount-responsive (published age curve)."""
    return {"13-17": 1.05, "18-24": 1.15, "25-34": 1.0,
            "35-44": 0.9, "45+": 0.8}[calibration.age_bracket(age)]


def _acceptance(offer: Dict, row: pd.Series, churn_prob: float) -> float:
    """Assumed acceptance probability for one offer / player."""
    is_spender = float(row.get("total_spend_usd", 0) or 0) > 0
    avg_purchase = float(row.get("avg_purchase_value", 0) or 0)
    age = int(row.get("age", 24) or 24)
    segment = str(row.get("spender_segment", "Non-spender"))

    base = (offer["base_accept_spender"] if is_spender
            else offer["base_accept_nonspender"])

    if offer["kind"] == "discount":
        mod = (0.6 + 0.8 * churn_prob) * _age_factor(age)   # at-risk & young
    elif offer["kind"] in ("premium", "spender"):
        # Premium/bundle lands with high-value spenders whose typical purchase
        # is near the offer's price point.
        seg_boost = {"Whale": 2.2, "Dolphin": 1.5}.get(segment, 0.6)
        fit = 1.0 if avg_purchase >= offer["price_point"] * 0.5 else 0.5
        mod = seg_boost * fit
    elif offer["kind"] == "engagement":
        mod = 0.7 + 0.6 * churn_prob                        # broad retention nudge
    else:
        mod = 0.0
    return float(min(base * mod, 0.95))


def _offer_ev(offer: Dict, row: pd.Series, churn_prob: float, value: float) -> Dict:
    accept = _acceptance(offer, row, churn_prob)
    churn_reduction = churn_prob * offer["retention_lift"] * accept
    retained_value = churn_reduction * value * HORIZON_MONTHS
    offer_profit = accept * offer["margin"]
    return {
        "offer_id": offer["id"],
        "offer_name": offer["name"],
        "acceptance": round(accept, 3),
        "expected_retained_value": round(retained_value, 2),
        "expected_offer_profit": round(offer_profit, 2),
        "expected_value": round(retained_value + offer_profit, 2),
    }


def _rationale(row: pd.Series, churn_prob: float, best: Dict) -> str:
    seg = str(row.get("spender_segment", "Non-spender"))
    risk = "high" if churn_prob >= 0.7 else "medium" if churn_prob >= 0.4 else "low"
    if best["offer_id"] == "none":
        return f"{risk.title()} churn risk, low expected uplift — no offer is +EV."
    return (f"{risk.title()} churn risk ({churn_prob:.0%}) + {seg} segment → "
            f"'{best['offer_name']}' maximises expected retained revenue.")


def recommend_offer(row: pd.Series, churn_prob: float) -> Dict:
    """Pick the highest expected-value retention offer for a player."""
    value = estimate_monthly_value(row)
    breakdown = [_offer_ev(o, row, churn_prob, value) for o in OFFERS]
    breakdown.sort(key=lambda d: d["expected_value"], reverse=True)
    best = breakdown[0]
    return {
        "estimated_monthly_value": round(value, 2),
        "revenue_at_risk": round(churn_prob * value * HORIZON_MONTHS, 2),
        "recommended_offer_id": best["offer_id"],
        "recommended_offer": best["offer_name"],
        "expected_value_of_offer": best["expected_value"],
        "expected_acceptance": best["acceptance"],
        "rationale": _rationale(row, churn_prob, best),
        "offer_breakdown": breakdown,
    }


def recommend_batch(df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
    """Recommendation for a scored cohort; returns a tidy frame aligned to df."""
    recs = [recommend_offer(df.iloc[i], float(churn_probs[i])) for i in range(len(df))]
    return pd.DataFrame([
        {
            "estimated_monthly_value": r["estimated_monthly_value"],
            "revenue_at_risk": r["revenue_at_risk"],
            "recommended_offer": r["recommended_offer"],
            "expected_value_of_offer": r["expected_value_of_offer"],
        }
        for r in recs
    ], index=df.index)
