# Player Churn Prediction Dashboard — task shortcuts.
# Usage: `make setup`, `make train`, `make serve`, etc.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PORT    ?= 8000

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV):  ## Create the virtualenv
	python3 -m venv $(VENV)

.PHONY: setup
setup: $(VENV)  ## Create venv + install dependencies (run `brew install libomp` on macOS for XGBoost)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: simulate
simulate:  ## Generate the simulated Valorant player base (data/valorant_players.csv)
	$(PY) -m src.simulate_valorant

.PHONY: eda
eda:  ## Run exploratory data analysis (prints stats, writes charts to reports/)
	$(PY) -m src.eda

.PHONY: train
train:  ## Train LR vs XGBoost, save best model + metrics + importance plot
	$(PY) -m src.train

.PHONY: serve
serve:  ## Run the FastAPI service (PORT=8000 by default; docs at /docs)
	$(VENV)/bin/uvicorn app.api:app --reload --port $(PORT)

.PHONY: dashboard
dashboard:  ## Run the Streamlit dashboard
	$(VENV)/bin/streamlit run app/dashboard.py

.PHONY: all
all: simulate eda train  ## Full pipeline: simulate players -> EDA -> train

.PHONY: docker-up
docker-up:  ## Build & run API + dashboard via docker compose
	docker compose up --build

.PHONY: docker-down
docker-down:  ## Stop and remove the compose stack
	docker compose down

.PHONY: clean
clean:  ## Remove caches, model, and generated reports
	rm -rf **/__pycache__ .pytest_cache
	rm -f models/*.joblib reports/*.png reports/*.json
