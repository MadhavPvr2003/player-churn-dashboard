"""Generate a realistic *simulated* Valorant player base.

Why simulated: no public dataset has Valorant player-level spend, age, and churn
(Riot doesn't release it). So we build a synthetic cohort whose relationships are
grounded in two real sources rather than invented:

  1. Spend magnitudes & the whale pyramid  -> the Kaggle "Mobile Game In-App
     Purchases 2025" dataset (Minnow/Dolphin/Whale segments, real $ ranges).
  2. Age -> spend and age -> conversion curves -> published industry statistics
     in src/calibration.py (the Kaggle set's age-spend correlation was ~0.02,
     i.e. noise, so we do NOT trust it for demographics).

Design that keeps churn honest (no leakage): every player has a latent
`engagement_health`. It drives the OBSERVABLE features (sessions, session length,
etc.) AND the recency `days_since_last_match` — but each through independent
noise. Churn is defined from recency (inactive > 14 days); `days_since_last_match`
is therefore EXCLUDED from the model features. The model must predict imminent
inactivity from the engagement/spend/demographic profile, not read it off recency.

Run:  python -m src.simulate_valorant
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.calibration import (
    AGE_SPEND_PROFILE,
    age_bracket,
    purchase_rate,
)

RNG = np.random.default_rng(20260727)

REGIONS = {
    "NA": 0.28, "EU": 0.26, "APAC": 0.18, "BR": 0.10, "KR": 0.10, "LATAM": 0.08,
}
RANKS = ["Iron", "Bronze", "Silver", "Gold", "Platinum",
         "Diamond", "Ascendant", "Immortal", "Radiant"]

CHURN_INACTIVITY_DAYS = 14  # industry-standard "lapsed player" threshold

# Spend segments among *paying* players — proportions & $ ranges from the Kaggle
# mobile-IAP dataset. (median, sigma) parameterise a lognormal per segment.
SPEND_SEGMENTS = {
    "Minnow":  {"share": 0.841, "median": 10.0,   "sigma": 0.35},
    "Dolphin": {"share": 0.136, "median": 245.0,  "sigma": 0.30},
    "Whale":   {"share": 0.023, "median": 2620.0, "sigma": 0.35},
}

# Valorant scaling: cosmetic-only monetisation converts a minority. This pulls
# the published (spending-inclined) conversion rates down to a realistic overall
# paying share (~30%).
VALORANT_CONVERSION_SCALE = 0.55


def _age_spend_multiplier(ages: np.ndarray) -> np.ndarray:
    """Per-player spend multiplier from published age->spend (peak at 18-24)."""
    peak = max(v[2] for v in AGE_SPEND_PROFILE.values())  # $21.50 (18-24)
    return np.array([AGE_SPEND_PROFILE[age_bracket(int(a))][2] / peak
                     for a in ages])


def simulate(n: int = 25000) -> pd.DataFrame:
    # --- Demographics --------------------------------------------------------
    age = np.clip(np.round(RNG.normal(24, 7, n)), 13, 54).astype(int)
    region = RNG.choice(list(REGIONS), size=n, p=list(REGIONS.values()))
    tenure_days = np.clip(RNG.exponential(260, n), 3, 1600).astype(int)

    # --- Latent engagement health (drives activity AND churn) ---------------
    health = np.clip(RNG.beta(2.2, 2.4, n), 0.01, 0.99)

    # --- Engagement features (functions of health with moderate noise) ------
    # Moderate noise keeps churn predictable-but-not-trivial from these signals.
    monthly_hours = health * RNG.uniform(3, 22, n)
    hours_played = np.round(monthly_hours * (tenure_days / 30.0)
                            * RNG.uniform(0.8, 1.15, n), 1)
    matches_played = np.round(hours_played * RNG.uniform(1.4, 2.2, n)).astype(int)
    sessions_per_week = np.clip(
        np.round(health * 18 * RNG.uniform(0.7, 1.2, n)), 0, 40).astype(int)
    avg_session_minutes = np.clip(
        np.round(20 + health * 90 + RNG.normal(0, 9, n)), 5, 240).astype(int)
    account_level = np.clip(
        np.round(hours_played * RNG.uniform(0.4, 0.9, n)) + 1, 1, 600).astype(int)

    # Rank tied to hours (skill ~ time invested) with noise.
    rank_score = np.argsort(np.argsort(
        hours_played * RNG.uniform(0.6, 1.4, n))) / n
    rank_idx = np.clip((rank_score * len(RANKS)).astype(int), 0, len(RANKS) - 1)
    rank = np.array(RANKS)[rank_idx]

    # --- Churn & recency -----------------------------------------------------
    # Churn probability is a logistic function of engagement health: the
    # threshold controls the churn *rate* (~30%) and the sharpness controls how
    # *separable* churners are. Sharpness is deliberately moderate (~6) so the
    # health-derived features predict churn well but not perfectly (target
    # AUC ~0.80), i.e. a realistic, non-leaky signal.
    CHURN_HEALTH_THRESHOLD = 0.35
    CHURN_SHARPNESS = 6.0
    p_churn = 1.0 / (1.0 + np.exp(-CHURN_SHARPNESS * (CHURN_HEALTH_THRESHOLD - health)))
    churn = (RNG.random(n) < p_churn).astype(int)
    # Recency is consistent with the label (churners are the lapsed ones). It is
    # EXCLUDED from features, so it never leaks — it's kept only for EDA/context.
    days_since_last_match = np.where(
        churn == 1,
        np.round(CHURN_INACTIVITY_DAYS + 1 + RNG.exponential(22, n)),
        np.round(RNG.uniform(0, CHURN_INACTIVITY_DAYS, n)),
    ).clip(0, 365).astype(int)

    # --- Monetisation --------------------------------------------------------
    p_spend = np.array([purchase_rate(int(a)) for a in age]) \
        * VALORANT_CONVERSION_SCALE * (0.5 + health)
    is_spender = RNG.random(n) < np.clip(p_spend, 0, 0.95)

    seg_names = list(SPEND_SEGMENTS)
    seg_share = np.array([SPEND_SEGMENTS[s]["share"] for s in seg_names])
    # Bias segment draw toward whales for highly engaged players.
    segment = np.empty(n, dtype=object)
    segment[:] = "Non-spender"
    spender_idx = np.where(is_spender)[0]
    for i in spender_idx:
        w = seg_share.copy()
        w[2] *= (0.5 + 2.0 * health[i])   # whales skew highly engaged
        w[1] *= (0.7 + health[i])
        w = w / w.sum()
        segment[i] = RNG.choice(seg_names, p=w)

    age_mult = _age_spend_multiplier(age)
    total_spend = np.zeros(n)
    num_purchases = np.zeros(n, dtype=int)
    for i in spender_idx:
        seg = SPEND_SEGMENTS[segment[i]]
        base = RNG.lognormal(np.log(seg["median"]), seg["sigma"])
        total_spend[i] = round(base * age_mult[i], 2)
        # Purchase count scales with segment size and tenure.
        cadence = {"Minnow": 2, "Dolphin": 8, "Whale": 22}[segment[i]]
        num_purchases[i] = max(1, RNG.poisson(cadence * (tenure_days[i] / 365.0)))

    avg_purchase_value = np.where(
        num_purchases > 0, np.round(total_spend / np.maximum(num_purchases, 1), 2), 0.0)

    owns_battlepass = (
        RNG.random(n) < np.clip(0.15 + 0.5 * health + 0.2 * is_spender, 0, 0.95)
    ).astype(int)

    # Purchase recency: spenders lapse roughly with activity; non-spenders get a
    # sentinel (no purchases). Kept independent enough to not leak the label.
    days_since_last_purchase = np.where(
        is_spender,
        np.round(RNG.exponential(10 + (1 - health) * 40, n)
                 + RNG.normal(0, 5, n)).clip(0, 400),
        999,  # sentinel = never purchased
    ).astype(int)

    df = pd.DataFrame({
        "PlayerID": np.arange(1, n + 1),
        "Age": age,
        "Region": region,
        "Rank": rank,
        "AccountLevel": account_level,
        "DaysSinceSignup": tenure_days,
        "HoursPlayed": hours_played,
        "MatchesPlayed": matches_played,
        "SessionsPerWeek": sessions_per_week,
        "AvgSessionMinutes": avg_session_minutes,
        "DaysSinceLastMatch": days_since_last_match,   # -> label only, not a feature
        "OwnsBattlepass": owns_battlepass,
        "SpenderSegment": segment,
        "NumPurchases": num_purchases,
        "TotalSpendUSD": total_spend.round(2),
        "AvgPurchaseValue": avg_purchase_value,
        "DaysSinceLastPurchase": days_since_last_purchase,
        "Churned": churn,
    })
    return df


if __name__ == "__main__":
    config.DATA_DIR.mkdir(exist_ok=True)
    df = simulate()
    df.to_csv(config.RAW_CSV, index=False)
    print(f"Wrote {len(df):,} simulated Valorant players -> {config.RAW_CSV}")
    print(f"Churn rate: {df.Churned.mean():.1%}")
    print(f"Paying share: {(df.SpenderSegment != 'Non-spender').mean():.1%}")
    rev = df.groupby('SpenderSegment').TotalSpendUSD.sum()
    print("Revenue by segment:\n", (rev / rev.sum() * 100).round(1))
