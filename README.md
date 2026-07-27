# 🎯 Valorant — Player Churn & Night Market Pricing Engine

Predict which Valorant players are about to lapse, then recommend the **retention
offer** — a personalised Night Market discount, battlepass nudge, or bundle —
that maximises **expected retained revenue**.

> **Problem → Approach → Result** — an end-to-end ML system: simulated-but-
> grounded data, a temporal churn model, and a churn-driven pricing engine,
> served through a FastAPI backend and a Streamlit dashboard.

---

## Problem

In a free-to-play game, a **tiny share of players drives most revenue** (here the
top ~2% "whales" generate ~60% of spend) and retention is the biggest lever on
lifetime value. Riot's Valorant already ships a personalised-discount feature —
the **Night Market** — but a random discount is a blunt instrument. The question
this project answers:

> *Given a player's churn risk, spending history, and age, which offer should we
> show them to keep them playing and paying — and what revenue is at risk if we
> do nothing?*

### An honest note on data

No public dataset has Valorant player-level **spend, age, and churn** (Riot
doesn't release it — it's PII + proprietary revenue). So the player base is
**simulated**, but its relationships are grounded in real sources rather than
invented:

| Signal | Grounded in |
|---|---|
| Spend magnitudes & the whale pyramid (Minnow/Dolphin/Whale) | Kaggle *Mobile Game In-App Purchases 2025* dataset |
| Age → spend and age → conversion curves | Published 2025 industry statistics (`src/calibration.py`) |
| Churn definition | Industry-standard **14-day inactivity** |

*(The Kaggle set's own age↔spend correlation was **0.02** — noise — so its
demographics were discarded and sourced from published stats instead. That
finding is documented in the code.)*

## Approach

1. **Simulate** (`src/simulate_valorant.py`) — 25k players with a latent
   `engagement_health` that drives the observable features **and** churn through
   *independent* noise, so churn is predictable but **not leaked**.
2. **Temporal churn label** — `churn = no match in > 14 days`. The recency that
   *defines* the label (`days_since_last_match`) is **excluded** from the model
   features — the model must predict lapse from behaviour, not read it off.
3. **Feature engineering** (`src/features.py`) — session frequency × duration,
   spend-per-hour, purchases-per-month, match cadence, and more.
4. **Modelling** (`src/train.py`) — LogisticRegression vs XGBoost in an sklearn
   `Pipeline`, class-imbalance handled, evaluated on the **churn class**
   (ROC-AUC / precision / recall / F1).
5. **Night Market pricing engine** (`src/pricing.py`) — for each player, pick the
   offer that maximises **expected value** = *retained revenue* (churn × value ×
   lift) + *offer profit*, using churn risk, real spend, and age.
6. **Serve** (`app/api.py`) — FastAPI `/predict` and `/recommend`.
7. **Dashboard** (`app/dashboard.py`) — players ranked by **revenue at risk**,
   offer-mix and revenue-by-segment charts, a live single-player scorer, and a
   **CSV batch scorer** (upload a roster → download everyone scored + offered).

## Result

**Churn model** — 25,000 players, 36% churn rate (held-out 20% test set, churn
class):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|:--:|:--:|:--:|:--:|
| **LogisticRegression** (shipped) | **0.79** | 0.58 | **0.73** | 0.65 |
| XGBoost | 0.78 | 0.58 | 0.72 | 0.64 |

**Headline: ROC-AUC 0.79, catching 73% of lapsing players.** That number is
deliberately *not* 0.99 — because churn is a genuine temporal signal with a
leakage-free label, 0.79 is what an honest behavioural churn model looks like
(and it beats the ~0.5 you'd get from guessing). Top drivers: session frequency
and session length (see `reports/feature_importance.png`).

**Pricing engine** — translates each score into an action. Examples:

| Player | Churn | Rev. at risk | Recommended offer |
|---|:--:|:--:|---|
| At-risk young **whale** | 78% | $783 | Night Market **40% win-back** |
| Healthy **dolphin** | 8% | $9 | **Bonus VP upsell** (no discount) |
| At-risk older **minnow** | 84% | $9 | Night Market **15% nudge** |

Note it *doesn't* discount a happy spender, and spends its deepest discounts
where the most revenue is at risk — the whole point of targeting.

> ⚠️ The pricing engine's acceptance rates and retention lifts are **business
> assumptions** (no offer/experiment log exists to learn them) — centralised in
> `src/pricing.py` and meant to be replaced with A/B-test or uplift-model
> results. The *framework* (value × churn × lift optimisation) is the real part.

---

## Project structure

```
player-churn-dashboard/
├── data/                    # simulated CSV + Kaggle reference set (git-ignored)
├── src/
│   ├── config.py            # paths, schema, churn definition, risk tiers
│   ├── calibration.py       # real published age→spend / whale stats (sourced)
│   ├── simulate_valorant.py # generates the grounded synthetic player base
│   ├── features.py          # load + churn label + feature engineering (shared)
│   ├── eda.py               # exploratory data analysis
│   ├── train.py             # LR vs XGBoost, saves best model + importance plot
│   └── pricing.py           # Night Market retention-offer engine
├── app/
│   ├── api.py               # FastAPI /predict + /recommend
│   └── dashboard.py         # Streamlit dashboard
├── models/  reports/        # trained model, metrics.json, charts (git-ignored)
├── Dockerfile  docker-compose.yml  Makefile  requirements.txt
```

---

## Running it locally

Every step has a `make` shortcut (`make help` lists them):

```bash
make setup        # create .venv + install deps   (macOS+XGBoost: brew install libomp)
make all          # simulate players -> EDA -> train
make serve        # FastAPI at http://localhost:8000/docs   (PORT=8000 by default)
make dashboard    # Streamlit dashboard             (second terminal)
```

Point the dashboard at the API if it's on another port:

```bash
CHURN_API_URL=http://localhost:8010 make dashboard
```

Example API call:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"age":20,"region":"NA","rank":"Immortal","account_level":300,
       "days_since_signup":500,"hours_played":400,"matches_played":800,
       "sessions_per_week":1,"avg_session_minutes":25,"owns_battlepass":1,
       "spender_segment":"Whale","num_purchases":30,"total_spend_usd":2800,
       "avg_purchase_value":93,"days_since_last_purchase":40}'
# -> churn 0.78, revenue_at_risk $783, recommended "Night Market — 40% off (win-back)"
```

### Run both with Docker

```bash
docker compose up --build      # or: make docker-up
# API       -> http://localhost:8000/docs
# Dashboard -> http://localhost:8501
```

One image backs both services; the player base is regenerated inside the image,
and the dashboard is wired to the API via `CHURN_API_URL`.

---

## Deploying

- **API → Render**: Web Service · build `pip install -r requirements.txt` ·
  start `uvicorn app.api:app --host 0.0.0.0 --port $PORT` · commit the model.
- **Dashboard → Hugging Face Spaces**: Streamlit Space · app file
  `app/dashboard.py` · set `CHURN_API_URL` (your Render URL) as a secret.
