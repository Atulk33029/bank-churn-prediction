"""Train a Random Forest churn classifier, evaluate it, and save all artifacts."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (FIGURES_DIR, METRICS_PATH, MODEL_PATH, POWERBI_EXPORT,
                    RANDOM_STATE, TEST_SIZE)
from data_prep import build_features, load_raw

sns.set_theme(style="whitegrid")


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def score(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def plot_confusion(y_test, pred):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest — confusion matrix")
    fig.savefig(FIGURES_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_test, proba, auc):
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2.5, color="#D1495B", label=f"Random Forest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "07_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, columns):
    imp = (pd.Series(model.feature_importances_, index=columns)
           .sort_values(ascending=False).head(12).reset_index())
    imp.columns = ["feature", "importance"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp, y="feature", x="importance", hue="feature",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Top drivers of customer churn")
    ax.set_ylabel("")
    fig.savefig(FIGURES_DIR / "08_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp


def main():
    raw = load_raw()
    X, y = build_features(raw)
    print(f"Dataset: {len(X):,} records, {X.shape[1]} features, churn rate {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    results = []
    trained = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(score(name, model, X_test, y_test))

    table = pd.DataFrame(results).set_index("model").round(4)
    print("Model comparison (hold-out test set)")
    print(table.to_string(), "\n")

    best = trained["Random Forest"]
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    cv = cross_val_score(best, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-fold CV accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})\n")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

    plot_confusion(y_test, pred)
    plot_roc(y_test, proba, auc)
    imp = plot_importance(best, X.columns)
    print("Top 5 churn drivers:")
    print(imp.head().to_string(index=False), "\n")

    joblib.dump({"model": best, "columns": X.columns.tolist()}, MODEL_PATH)

    metrics = {
        "comparison": results,
        "best_model": "Random Forest",
        "cv_accuracy_mean": cv.mean(),
        "cv_accuracy_std": cv.std(),
        "top_features": imp.to_dict("records"),
        "n_records": int(len(X)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))

    # Full scored dataset for the Power BI dashboard
    export = raw.copy()
    export["ChurnProbability"] = best.predict_proba(X)[:, 1].round(4)
    export["PredictedChurn"] = (export["ChurnProbability"] >= 0.5).astype(int)
    export["RiskBand"] = pd.cut(export["ChurnProbability"],
                                bins=[-0.01, 0.3, 0.6, 1.0],
                                labels=["Low", "Medium", "High"])
    export["AgeGroup"] = pd.cut(export["Age"], bins=[17, 30, 40, 50, 60, 100],
                                labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    export.to_csv(POWERBI_EXPORT, index=False)

    print(f"Model saved      -> {MODEL_PATH}")
    print(f"Metrics saved    -> {METRICS_PATH}")
    print(f"Power BI export  -> {POWERBI_EXPORT}")
    print(f"Charts saved     -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
