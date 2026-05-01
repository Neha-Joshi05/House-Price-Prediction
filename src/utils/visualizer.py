"""
visualizer.py
-------------
All matplotlib / seaborn plots for EDA and model evaluation.
Saves all figures to outputs/plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PLOTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, filename: str):
    path = PLOTS_PATH / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Saved → {path}")


# ── EDA Plots ────────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("House Price Distribution", fontsize=16, fontweight="bold")

    axes[0].hist(df["price"] / 1_000, bins=50,
                 color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Price ($ thousands)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Raw Price")

    axes[1].hist(np.log1p(df["price"]), bins=50,
                 color="#55A868", edgecolor="white")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_title("Log-Transformed Price")

    fig.tight_layout()
    _save(fig, "01_price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask    = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    _save(fig, "02_correlation_heatmap.png")


def plot_feature_vs_price(df: pd.DataFrame):
    features = ["area_sqft", "bedrooms", "bathrooms",
                "age_years", "distance_cbd_km", "school_rating"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Key Features vs. Price", fontsize=16, fontweight="bold")

    for ax, col in zip(axes.flatten(), features):
        ax.scatter(df[col], df["price"] / 1_000,
                   alpha=0.3, s=15, color="#4C72B0")
        ax.set_xlabel(col)
        ax.set_ylabel("Price ($K)")
        ax.set_title(col)

    fig.tight_layout()
    _save(fig, "03_feature_vs_price.png")


def plot_categorical_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Categorical Features vs. Price", fontsize=15, fontweight="bold")

    order = ["unfurnished", "semi-furnished", "furnished"]
    sns.boxplot(data=df, x="furnished", y="price",
                order=order, palette="Set2", ax=axes[0])
    axes[0].set_title("Furnished Status")
    axes[0].set_ylabel("Price ($)")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1_000:.0f}K"))

    sns.boxplot(data=df, x="location_tier", y="price",
                palette="Set3", ax=axes[1])
    axes[1].set_title("Location Tier  (1=Premium → 4=Budget)")
    axes[1].set_ylabel("")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1_000:.0f}K"))

    fig.tight_layout()
    _save(fig, "04_categorical_boxplots.png")


# ── Model Evaluation Plots ───────────────────────────────────

def plot_actual_vs_predicted(trained_models: dict, X_test, y_test):
    n     = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    fig.suptitle("Actual vs. Predicted Prices", fontsize=15, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        ax.scatter(y_test / 1_000, y_pred / 1_000,
                   alpha=0.35, s=18, color="#4C72B0")
        mn = min(y_test.min(), y_pred.min()) / 1_000
        mx = max(y_test.max(), y_pred.max()) / 1_000
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual ($K)")
        ax.set_ylabel("Predicted ($K)")
        ax.set_title(name)
        ax.legend()

    fig.tight_layout()
    _save(fig, "05_actual_vs_predicted.png")


def plot_residuals(trained_models: dict, X_test, y_test):
    n     = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    fig.suptitle("Residual Plots", fontsize=15, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred    = model.predict(X_test)
        residuals = y_test.values - y_pred
        ax.scatter(y_pred / 1_000, residuals / 1_000,
                   alpha=0.35, s=18, color="#C44E52")
        ax.axhline(0, color="black", lw=1.5, ls="--")
        ax.set_xlabel("Predicted ($K)")
        ax.set_ylabel("Residual ($K)")
        ax.set_title(name)

    fig.tight_layout()
    _save(fig, "06_residuals.png")


def plot_model_comparison(report_df: pd.DataFrame):
    metrics = ["MAE ($)", "RMSE ($)", "R² Score", "MAPE (%)"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes.flatten(), metrics):
        vals = report_df[metric]
        bars = ax.barh(report_df.index, vals, color=colors[:len(report_df)])
        ax.set_xlabel(metric)
        ax.set_title(metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 0.98,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,.2f}", va="center", ha="right",
                    color="white", fontweight="bold", fontsize=9)

    fig.tight_layout()
    _save(fig, "07_model_comparison.png")


def plot_feature_importance(model, feature_names: list, model_name: str = "XGBoost"):
    if not hasattr(model, "feature_importances_"):
        print(f"   ⚠️  {model_name} has no feature_importances_ — skipping.")
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    imp.plot(kind="barh", color="#4C72B0", edgecolor="white", ax=ax)
    ax.set_title(f"{model_name} — Feature Importances",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    _save(fig, "08_feature_importance.png")