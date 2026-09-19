"""Exploratory data analysis: saves Seaborn charts to reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, TARGET
from data_prep import load_raw, clean, add_features

sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = {0: "#4C9F70", 1: "#D1495B"}


def save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(FIGURES_DIR.parents[1])}")


def churn_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
    total = len(df)
    for p in ax.patches:
        ax.annotate(f"{p.get_height() / total:.1%}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Churn distribution")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_xlabel("")
    save(fig, "01_churn_distribution")


def churn_by_category(df):
    cats = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard", "Tenure"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, cats):
        rate = df.groupby(col)[TARGET].mean().reset_index()
        sns.barplot(data=rate, x=col, y=TARGET, hue=col, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Where does churn concentrate?", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "02_churn_by_category")


def numeric_distributions(df):
    nums = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.flat, nums):
        sns.kdeplot(data=df, x=col, hue=TARGET, fill=True, alpha=.35,
                    palette=PALETTE, common_norm=False, ax=ax)
        ax.set_title(f"{col} by churn status")
    fig.tight_layout()
    save(fig, "03_numeric_distributions")


def age_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Geography", y="Age", hue=TARGET, palette=PALETTE, ax=ax)
    ax.set_title("Age vs geography, split by churn")
    ax.legend(title="Churned", labels=["No", "Yes"])
    save(fig, "04_age_geography_box")


def correlation(df):
    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, "05_correlation_heatmap")


def main():
    df = add_features(clean(load_raw()))
    print(f"Loaded {len(df):,} customer records | churn rate {df[TARGET].mean():.2%}")
    churn_balance(df)
    churn_by_category(df)
    numeric_distributions(df)
    age_boxplot(df)
    correlation(df)
    print("\nEDA complete. Charts are in reports/figures/")


if __name__ == "__main__":
    main()
