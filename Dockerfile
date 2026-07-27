# Shared image for both the API and the dashboard.
# The service (uvicorn vs streamlit) is chosen by the command in
# docker-compose.yml, so one build serves both.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime XGBoost needs on Linux (the Linux equivalent
# of `brew install libomp` on macOS).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and the trained model (run `make train` before building so
# models/churn_model.joblib exists).
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Generate the simulated player base inside the image (deterministic) so the
# dashboard has a cohort to score. The API scores request payloads and needs
# only the model.
RUN python -m src.simulate_valorant

EXPOSE 8000 8501

# Default command runs the API; the dashboard service overrides it in compose.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
