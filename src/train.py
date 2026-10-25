"""Train and evaluate baseline classifiers on a built-in scikit-learn dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
REPORTS_DIR = Path("reports")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the included dataset and return features, target, and class names."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y, list(dataset.target_names)


def build_models() -> dict[str, object]:
    """Create models; scaling is fitted only on each training fold."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def evaluate_model(model, X_test, y_test, class_names: list[str]) -> dict[str, float]:
    """Return common classification metrics for the held-out test set."""
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, class_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Classes: {class_names}")
    print("\n5-fold cross-validation on training data:")

    results: dict[str, dict[str, float]] = {}
    fitted_models = build_models()
    comparison_rows = []

    for name, model in fitted_models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="f1", n_jobs=None
        )
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, class_names)
        metrics["cv_f1_mean"] = float(np.mean(cv_scores))
        metrics["cv_f1_std"] = float(np.std(cv_scores))
        results[name] = metrics
        comparison_rows.append({"model": name, **metrics})

        print(f"\n{name}")
        print(f"  CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        print(f"  Test accuracy: {metrics['accuracy']:.3f}")
        print(f"  Test precision: {metrics['precision']:.3f}")
        print(f"  Test recall: {metrics['recall']:.3f}")
        print(f"  Test F1: {metrics['f1']:.3f}")
        print(f"  Test ROC-AUC: {metrics['roc_auc']:.3f}")

    best_name = max(results, key=lambda name: results[name]["cv_f1_mean"])
    best_model = fitted_models[best_name]
    print(f"\nSelected by mean CV F1: {best_name}")
    print("\nClassification report on held-out test set:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=class_names))

    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"selected_model_by_cv_f1": best_name, "models": results},
            file,
            indent=2,
        )
    pd.DataFrame(comparison_rows).to_csv(
        REPORTS_DIR / "model_comparison.csv", index=False
    )

    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=class_names, cmap="Blues"
    )
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC curve — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=160)
    plt.close()

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    top_indices = np.argsort(importance.importances_mean)[-10:]
    plt.figure(figsize=(9, 6))
    plt.barh(
        X.columns[top_indices],
        importance.importances_mean[top_indices],
        xerr=importance.importances_std[top_indices],
    )
    plt.xlabel("Mean decrease in F1 after permutation")
    plt.title(f"Top permutation importances — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "permutation_importance.png", dpi=160)
    plt.close()

    print(f"\nReports saved to: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
