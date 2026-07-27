"""Real-world calibration constants for the simulated player base.

The player data in this project is synthetic (no public Valorant/real-money
player data exists), but the *relationships* between age, spending, and
retention are grounded in published industry statistics rather than invented.
Sources are cited inline; swap these for your own analytics when available.

Sources:
  - In-game purchase spending habits by age, 2025:
    https://coopboardgames.com/statistics/in-game-purchase-spending-habit-statistics/
  - In-game purchase statistics 2026 (participation, ARPPU, whale share):
    https://sqmagazine.co.uk/in-game-purchases-statistics/
"""
from __future__ import annotations

# Average monthly spend (USD) and share of players who spend, by age bracket.
# From published 2025 spending-by-age figures. Younger players both spend more
# and convert at higher rates.
AGE_SPEND_PROFILE = {
    #  bracket:    (age_lo, age_hi, monthly_spend_usd, purchase_rate)
    "13-17": (13, 17, 9.80, 0.48),
    "18-24": (18, 24, 21.50, 0.72),
    "25-34": (25, 34, 19.30, 0.68),
    "35-44": (35, 44, 14.60, 0.55),
    "45+":   (45, 65, 7.90, 0.31),
}

# Revenue concentration ("whales"): a tiny minority drives most spend.
REVENUE_CONCENTRATION = {
    "never_spend_share": 0.65,     # 60–70% of F2P players never pay
    "top_5pct_revenue_share": 0.50,  # top 5% of spenders -> ~50% of revenue
    "top_1pct_revenue_share": 0.30,  # top 1% -> ~30% of revenue
}

# Benchmark annual revenue per paying user for competitive titles (Valorant-like
# tactical shooter sits near/above this). Used to sanity-check simulated spend.
ARPPU_ANNUAL_USD = {
    "global_avg": 87.0,
    "us_avg": 112.0,
    "competitive_titles": 150.0,
}

# Casual paying players make ~1–3 purchases/month; used to shape purchase counts.
CASUAL_MONTHLY_PURCHASES = (1, 3)


def age_bracket(age: int) -> str:
    """Return the spending bracket key for an age."""
    for key, (lo, hi, _spend, _rate) in AGE_SPEND_PROFILE.items():
        if lo <= age <= hi:
            return key
    return "45+" if age > 45 else "13-17"


def expected_monthly_spend(age: int) -> float:
    """Bracket-average monthly spend for an age (before per-player variation)."""
    return AGE_SPEND_PROFILE[age_bracket(age)][2]


def purchase_rate(age: int) -> float:
    """Probability a player of this age is a spender at all."""
    return AGE_SPEND_PROFILE[age_bracket(age)][3]
