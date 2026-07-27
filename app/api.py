"""FastAPI service: churn prediction + Night Market retention-offer recommendation.

Run:  uvicorn app.api:app --reload --port 8000

Endpoints
---------
GET  /health          -> service + model status
POST /predict         -> a player's churn probability + risk tier
POST /recommend       -> churn + the best retention offer (revenue-at-risk, EV)
POST /predict/batch   -> list of players
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Make `src` importable when launched from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.features import engineer_features
from src.pricing import recommend_offer

app = FastAPI(
    title="Valorant Churn & Night Market Pricing API",
    description="Predicts player churn and recommends a retention offer.",
    version="2.0.0",
)

_model = None


def get_model():
    """Lazy-load the trained pipeline so import never fails if it's missing."""
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(f"Model not found at {config.MODEL_PATH}. "
                        "Run `python -m src.train` first."),
            )
        _model = joblib.load(config.MODEL_PATH)
    return _model


class Player(BaseModel):
    """Raw player stats — one row of the Valorant player base."""
    age: int = Field(24, ge=13, le=80, examples=[21])
    region: str = Field("NA", examples=["NA"])
    rank: str = Field("Gold", examples=["Platinum"])
    account_level: int = Field(20, ge=1, examples=[85])
    days_since_signup: int = Field(180, ge=1, examples=[420])
    hours_played: float = Field(50.0, ge=0, examples=[220.5])
    matches_played: int = Field(100, ge=0, examples=[430])
    sessions_per_week: float = Field(3.0, ge=0, examples=[9])
    avg_session_minutes: float = Field(60.0, ge=0, examples=[95])
    owns_battlepass: int = Field(0, ge=0, le=1, examples=[1])
    spender_segment: str = Field("Non-spender", examples=["Dolphin"])
    num_purchases: int = Field(0, ge=0, examples=[12])
    total_spend_usd: float = Field(0.0, ge=0, examples=[240.0])
    avg_purchase_value: float = Field(0.0, ge=0, examples=[20.0])
    days_since_last_purchase: int = Field(999, ge=0, examples=[15])


class PredictionOut(BaseModel):
    churn_probability: float
    risk_tier: str
    will_churn: bool


class RecommendationOut(PredictionOut):
    """Churn prediction plus the best retention offer to make."""
    estimated_monthly_value: float
    revenue_at_risk: float
    recommended_offer: str
    expected_value_of_offer: float
    expected_acceptance: float
    rationale: str
    offer_breakdown: List[dict]


def _score(df: pd.DataFrame):
    model = get_model()
    X = engineer_features(df)
    return model.predict_proba(X)[:, 1]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": config.MODEL_PATH.exists(),
        "model_path": str(config.MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionOut)
def predict(player: Player):
    df = pd.DataFrame([player.model_dump()])
    p = float(_score(df)[0])
    return PredictionOut(
        churn_probability=round(p, 4),
        risk_tier=config.risk_tier(p),
        will_churn=bool(p >= 0.5),
    )


@app.post("/recommend", response_model=RecommendationOut)
def recommend(player: Player):
    df = pd.DataFrame([player.model_dump()])
    p = float(_score(df)[0])
    rec = recommend_offer(df.iloc[0], p)
    return RecommendationOut(
        churn_probability=round(p, 4),
        risk_tier=config.risk_tier(p),
        will_churn=bool(p >= 0.5),
        **rec,
    )


@app.post("/predict/batch", response_model=List[PredictionOut])
def predict_batch(players: List[Player]):
    if not players:
        raise HTTPException(status_code=400, detail="Empty player list.")
    df = pd.DataFrame([p.model_dump() for p in players])
    proba = _score(df)
    return [
        PredictionOut(
            churn_probability=round(float(p), 4),
            risk_tier=config.risk_tier(float(p)),
            will_churn=bool(p >= 0.5),
        )
        for p in proba
    ]
