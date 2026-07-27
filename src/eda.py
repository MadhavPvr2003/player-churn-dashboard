"""Exploratory data analysis for the online-gaming behaviour dataset.

Run:  python -m src.eda

Prints schema, null counts, summary stats, and churn class balance, and saves a
couple of PNG charts to reports/ so the notebook-free workflow still produces
visual artifacts for the README.
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config
from src.features import derive_churn, load_raw


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run() -> None:
    try:
        df = load_raw()
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    _section("SHAPE & COLUMNS")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print("\nColumns:", list(df.columns))

    _section("DTYPES & INFO")
    df.info()

    _section("NULL / MISSING VALUES")
    nulls = df.isna().sum()
    if nulls.sum() == 0:
        print("No missing values in any column.")
    else:
        print(nulls[nulls > 0].sort_values(ascending=False))

    _section("NUMERIC SUMMARY STATS")
    print(df.describe(include=[float, int]).T)

    _section("CATEGORICAL VALUE COUNTS")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n-- {col} --")
        print(df[col].value_counts().head(10))

    _section("CHURN LABEL — CLASS BALANCE")
    print(f"Churn = no match played in > {config.CHURN_INACTIVITY_DAYS} days "
          "(temporal / inactivity definition).")
    if "days_since_last_match" in df.columns:
        print("\ndays_since_last_match summary (label source, EXCLUDED from features):")
        print(df["days_since_last_match"].describe())

    churn = derive_churn(df)
    counts = churn.value_counts().sort_index()
    rate = churn.mean()
    print(f"\nchurn=0 (retained): {counts.get(0, 0):,}")
    print(f"churn=1 (at-risk):  {counts.get(1, 0):,}")
    print(f"Churn rate: {rate:.1%}  "
          f"({'imbalanced' if rate < 0.35 or rate > 0.65 else 'roughly balanced'})")

    # Optional charts — skipped silently if matplotlib is unavailable.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config.REPORTS_DIR.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        counts.rename({0: "retained", 1: "at-risk"}).plot.bar(
            ax=ax, color=["#4C9F70", "#D1495B"])
        ax.set_title("Churn class balance")
        ax.set_ylabel("players")
        fig.tight_layout()
        fig.savefig(config.REPORTS_DIR / "churn_balance.png", dpi=120)

        num = df.select_dtypes(include=[float, int])
        if not num.empty:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(num.corr(), cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(num.columns)))
            ax.set_xticklabels(num.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(num.columns)))
            ax.set_yticklabels(num.columns, fontsize=7)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            ax.set_title("Numeric feature correlation")
            fig.tight_layout()
            fig.savefig(config.REPORTS_DIR / "correlation.png", dpi=120)
        print(f"\nCharts written to {config.REPORTS_DIR}/")
    except ImportError:
        print("\n(matplotlib not installed — skipped chart export)")


if __name__ == "__main__":
    run()
