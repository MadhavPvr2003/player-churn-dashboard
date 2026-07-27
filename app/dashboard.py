"""Valorant churn & Night Market pricing dashboard.

Run:  streamlit run app/dashboard.py

Shows a churn-ranked player table with recommended retention offers, revenue-at-
risk and offer-mix charts, and a single-player scorer that calls the FastAPI
/recommend endpoint (falling back to the local model + pricing engine if offline).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src import config
from src.features import engineer_features, load_raw, normalize_columns
from src.pricing import recommend_batch, recommend_offer

API_URL = os.getenv("CHURN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Valorant Churn & Pricing", page_icon="🎯", layout="wide")
TIER_COLORS = {"low": "#4C9F70", "medium": "#E9C46A", "high": "#FF4655"}  # Valorant red


@st.cache_resource
def load_model():
    import joblib
    return joblib.load(config.MODEL_PATH) if config.MODEL_PATH.exists() else None


@st.cache_data
def load_players() -> pd.DataFrame:
    raw = load_raw()
    return raw.sample(min(600, len(raw)), random_state=1).reset_index(drop=True)


@st.cache_data
def score_cohort(_model_id: int) -> pd.DataFrame:
    """Score the cohort and attach churn + Night Market recommendations."""
    model = load_model()
    players = load_players()
    proba = model.predict_proba(engineer_features(players))[:, 1]
    out = players.copy()
    out["churn_probability"] = proba.round(4)
    out["risk_tier"] = [config.risk_tier(p) for p in proba]
    recs = recommend_batch(players, proba)
    out = pd.concat([out, recs], axis=1)
    return out.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)


# --- UI ---------------------------------------------------------------------
st.title("🎯 Valorant — Churn & Night Market Pricing")
st.caption("Predict which players will lapse, then recommend the retention offer "
           "that maximises expected retained revenue. *Simulated player base — "
           "grounded in real spend distributions & published age→spend stats.*")

model = load_model()
if model is None:
    st.error("No trained model. Run `python -m src.train` first, then reload.")
    st.stop()

scored = score_cohort(id(model))

# KPI row.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players scored", f"{len(scored):,}")
c2.metric("High-risk", f"{(scored.risk_tier == 'high').sum():,}")
c3.metric("Revenue at risk", f"${scored.revenue_at_risk.sum():,.0f}")
c4.metric("Expected offer uplift", f"${scored.expected_value_of_offer.sum():,.0f}")

left, right = st.columns([3, 2])

with left:
    st.subheader("Players ranked by revenue at risk")
    tiers = st.multiselect("Risk tier", ["high", "medium", "low"],
                           default=["high", "medium"])
    view = scored[scored.risk_tier.isin(tiers)] if tiers else scored
    cols = [c for c in [
        "player_id", "age", "rank", "spender_segment", "total_spend_usd",
        "sessions_per_week", "churn_probability", "revenue_at_risk",
        "recommended_offer"] if c in view.columns]
    st.dataframe(
        view[cols].head(200), width="stretch", hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "churn", min_value=0.0, max_value=1.0, format="%.2f"),
            "revenue_at_risk": st.column_config.NumberColumn(
                "rev @ risk", format="$%.0f"),
            "total_spend_usd": st.column_config.NumberColumn("spend", format="$%.0f"),
        },
    )

with right:
    st.subheader("Recommended offer mix")
    mix = scored.recommended_offer.value_counts()
    st.bar_chart(mix, horizontal=True, color="#FF4655")
    st.subheader("Revenue at risk by segment")
    seg = scored.groupby("spender_segment").revenue_at_risk.sum().sort_values(ascending=False)
    st.bar_chart(seg, color="#FF4655")

st.divider()

# --- Single-player scorer ---
st.subheader("🔮 Score a player & recommend an offer")
with st.form("single"):
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", 13, 80, 21)
        region = st.selectbox("Region", ["NA", "EU", "APAC", "KR", "BR", "LATAM"])
        rank = st.selectbox("Rank", ["Iron", "Bronze", "Silver", "Gold", "Platinum",
                                     "Diamond", "Ascendant", "Immortal", "Radiant"], index=3)
        account_level = st.number_input("Account level", 1, 600, 85)
        days_since_signup = st.number_input("Days since signup", 1, 2000, 400)
    with b:
        hours_played = st.number_input("Hours played", 0.0, 5000.0, 220.0)
        matches_played = st.number_input("Matches played", 0, 10000, 430)
        sessions_per_week = st.number_input("Sessions / week", 0.0, 50.0, 2.0)
        avg_session_minutes = st.number_input("Avg session (min)", 0.0, 300.0, 40.0)
        owns_battlepass = st.selectbox("Owns battlepass", [0, 1])
    with c:
        spender_segment = st.selectbox("Spender segment",
                                       ["Non-spender", "Minnow", "Dolphin", "Whale"], index=2)
        num_purchases = st.number_input("Num purchases", 0, 500, 10)
        total_spend_usd = st.number_input("Total spend ($)", 0.0, 20000.0, 240.0)
        avg_purchase_value = st.number_input("Avg purchase ($)", 0.0, 5000.0, 24.0)
        days_since_last_purchase = st.number_input("Days since last purchase", 0, 999, 45)
    submitted = st.form_submit_button("Predict & recommend")

if submitted:
    payload = {
        "age": int(age), "region": region, "rank": rank,
        "account_level": int(account_level), "days_since_signup": int(days_since_signup),
        "hours_played": float(hours_played), "matches_played": int(matches_played),
        "sessions_per_week": float(sessions_per_week),
        "avg_session_minutes": float(avg_session_minutes),
        "owns_battlepass": int(owns_battlepass), "spender_segment": spender_segment,
        "num_purchases": int(num_purchases), "total_spend_usd": float(total_spend_usd),
        "avg_purchase_value": float(avg_purchase_value),
        "days_since_last_purchase": int(days_since_last_purchase),
    }
    result, source = None, None
    try:
        resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=3)
        resp.raise_for_status()
        result, source = resp.json(), "FastAPI /recommend"
    except Exception:
        df = pd.DataFrame([payload])
        prob = float(model.predict_proba(engineer_features(df))[:, 1][0])
        rec = recommend_offer(df.iloc[0], prob)
        result = {"churn_probability": prob, "risk_tier": config.risk_tier(prob), **rec}
        source = "local model (API offline)"

    prob, tier = result["churn_probability"], result["risk_tier"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Revenue at risk", f"${result['revenue_at_risk']:,.0f}")
    m3.markdown(
        f"<div style='padding:1rem;border-radius:8px;background:{TIER_COLORS[tier]};"
        f"color:white;text-align:center;font-size:1.3rem;font-weight:700'>"
        f"{tier.upper()} RISK</div>", unsafe_allow_html=True)
    st.success(f"**Recommended offer: {result['recommended_offer']}**  "
               f"·  expected value ${result['expected_value_of_offer']:,.2f}  "
               f"·  ~{result['expected_acceptance']:.0%} acceptance")
    st.caption(result.get("rationale", ""))
    with st.expander("Full offer breakdown (expected value of every offer)"):
        st.dataframe(pd.DataFrame(result["offer_breakdown"]), hide_index=True,
                     width="stretch")
    st.caption(f"Served via: {source}")

st.divider()

# --- Batch scoring: upload a roster CSV -------------------------------------
st.subheader("📤 Batch-score a roster (CSV upload)")
st.caption("Upload a CSV of players to score the whole roster and get a "
           "retention offer for each. Column names are matched flexibly; "
           "missing columns fall back to neutral defaults.")

# Downloadable template so users know the expected columns.
_template = load_players().drop(
    columns=[c for c in ["churned", "days_since_last_match"] if c in load_players().columns]
).head(10)
st.download_button("⬇️ Download a template CSV (10 sample players)",
                   _template.to_csv(index=False), "roster_template.csv", "text/csv")

uploaded = st.file_uploader("Upload roster CSV", type="csv")
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded, keep_default_na=False, na_values=[""])
        roster = normalize_columns(raw)
        proba = model.predict_proba(engineer_features(roster))[:, 1]
        recs = recommend_batch(roster, proba)
        result = roster.copy()
        result["churn_probability"] = proba.round(4)
        result["risk_tier"] = [config.risk_tier(p) for p in proba]
        result = pd.concat([result, recs], axis=1).sort_values(
            "revenue_at_risk", ascending=False).reset_index(drop=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Players scored", f"{len(result):,}")
        k2.metric("High-risk", f"{(result.risk_tier == 'high').sum():,}")
        k3.metric("Revenue at risk", f"${result.revenue_at_risk.sum():,.0f}")

        show = [c for c in ["player_id", "age", "rank", "spender_segment",
                            "total_spend_usd", "churn_probability", "risk_tier",
                            "revenue_at_risk", "recommended_offer"]
                if c in result.columns]
        st.dataframe(result[show], hide_index=True, width="stretch",
                     column_config={
                         "churn_probability": st.column_config.ProgressColumn(
                             "churn", min_value=0.0, max_value=1.0, format="%.2f"),
                         "revenue_at_risk": st.column_config.NumberColumn(
                             "rev @ risk", format="$%.0f"),
                     })
        st.download_button("⬇️ Download scored roster",
                           result.to_csv(index=False), "scored_roster.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not score that file: {exc}")
