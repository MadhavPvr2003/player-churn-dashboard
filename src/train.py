"""Train and compare a LogisticRegression baseline and an XGBoost model.

Run:  python -m src.train

Both models are wrapped in an sklearn Pipeline (preprocessing + estimator) so
the exact transform used in training is serialised with the model and reused
verbatim by the FastAPI service. Evaluation focuses on the churn (positive)
class — ROC-AUC, precision, recall, F1 — because the label is imbalanced and
accuracy would be misleading.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from src import config
from src.features import build_training_frame


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
        ]
    )


def plot_feature_importance(name: str, pipe: Pipeline, top_n: int = 15) -> None:
    """Save a horizontal bar chart of the winning model's top features.

    Uses standardised LogisticRegression coefficients (magnitude = influence,
    sign = direction) or XGBoost gain-based importances. Categorical one-hot
    columns keep their `cat__<col>_<value>` names so the chart is readable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipped importance plot)")
        return

    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
        subtitle = "standardised logistic-regression coefficients"
    else:
        values = clf.feature_importances_
        subtitle = "XGBoost gain importance"

    order = np.argsort(np.abs(values))[::-1][:top_n]
    names = [feat_names[i] for i in order][::-1]
    vals = [values[i] for i in order][::-1]
    colors = ["#D1495B" if v >= 0 else "#4C9F70" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_title(f"Top {top_n} churn drivers — {name}\n({subtitle})", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = config.REPORTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=120)
    print(f"Saved importance -> {out}")


def _evaluate(name, pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision_churn": float(precision_score(y_test, pred, zero_division=0)),
        "recall_churn": float(recall_score(y_test, pred, zero_division=0)),
        "f1_churn": float(f1_score(y_test, pred, zero_division=0)),
    }
    print(f"\n----- {name} -----")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Precision (churn): {metrics['precision_churn']:.4f}")
    print(f"Recall    (churn): {metrics['recall_churn']:.4f}")
    print(f"F1        (churn): {metrics['f1_churn']:.4f}")
    print(classification_report(y_test, pred,
                                target_names=["retained", "at-risk"],
                                zero_division=0))
    return metrics


def run() -> None:
    X, y, _ = build_training_frame()
    print(f"Loaded {len(X):,} players | churn rate = {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # --- Baseline: Logistic Regression ---
    logreg = Pipeline([
        ("prep", _preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)
    m_lr = _evaluate("LogisticRegression", logreg, X_test, y_test)

    # --- XGBoost ---
    from xgboost import XGBClassifier

    # scale_pos_weight balances the positive (churn) class.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / pos) if pos else 1.0

    xgb = Pipeline([
        ("prep", _preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    xgb.fit(X_train, y_train)
    m_xgb = _evaluate("XGBoost", xgb, X_test, y_test)

    # --- Pick the winner by ROC-AUC ---
    results = [(m_lr, logreg), (m_xgb, xgb)]
    best_metrics, best_pipe = max(results, key=lambda r: r[0]["roc_auc"])
    print(f"\n==> Best model: {best_metrics['model']} "
          f"(ROC-AUC {best_metrics['roc_auc']:.4f})")

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, config.MODEL_PATH)
    plot_feature_importance(best_metrics["model"], best_pipe)

    report = {
        "best_model": best_metrics["model"],
        "churn_rate": float(y.mean()),
        "n_players": int(len(X)),
        "metrics": {"logistic_regression": m_lr, "xgboost": m_xgb},
        "feature_columns": config.FEATURE_COLUMNS,
    }
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Saved model  -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    run()
