"""Train, compare, and persist fake-news classification models."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.config import (
    ASSETS_DIR,
    BEST_MODEL_PATH,
    MAX_DF,
    MAX_FEATURES,
    METRICS_PATH,
    MIN_DF,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_SEED,
    VECTORIZER_PATH,
    ensure_project_dirs,
)
from src.data_loader import load_dataset
from src.preprocessing import clean_text

_LABEL_NAMES = ["FAKE", "REAL"]


def _make_vectorizer() -> TfidfVectorizer:
    """Create the shared training and cross-validation TF-IDF settings."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )


def _make_estimators() -> dict[str, Any]:
    """Create the three requested classifiers with a fixed random seed."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=RANDOM_SEED), cv=3, method="sigmoid"
        ),
    }


def _model_slug(model_name: str) -> str:
    """Convert a display model name to a stable artifact filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _linear_weights(model: Any) -> np.ndarray | None:
    """Return model feature weights where the estimator exposes them."""
    if hasattr(model, "feature_log_prob_"):
        return np.asarray(model.feature_log_prob_[1] - model.feature_log_prob_[0])
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        estimator = getattr(
            calibrated[0],
            "estimator",
            getattr(calibrated[0], "base_estimator", None),
        )
        if estimator is not None and hasattr(estimator, "coef_"):
            return np.asarray(estimator.coef_).ravel()
    return None


def _top_indicative_words(
    model: Any, vectorizer: TfidfVectorizer, limit: int = 15
) -> dict[str, list[dict[str, float | str]]]:
    """Return the strongest feature terms for fake and real predictions."""
    weights = _linear_weights(model)
    if weights is None:
        return {"FAKE": [], "REAL": []}
    feature_names = vectorizer.get_feature_names_out()
    fake_indices = np.argsort(weights)[:limit]
    real_indices = np.argsort(weights)[-limit:][::-1]
    return {
        "FAKE": [
            {"word": str(feature_names[index]), "score": float(weights[index])}
            for index in fake_indices
        ],
        "REAL": [
            {"word": str(feature_names[index]), "score": float(weights[index])}
            for index in real_indices
        ],
    }


def _save_plots(
    results: dict[str, dict[str, Any]], labels: pd.Series
) -> None:
    """Save confusion-matrix, metric-comparison, and class-count charts."""
    for model_name, result in results.items():
        figure, axis = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=_LABEL_NAMES,
            yticklabels=_LABEL_NAMES,
            ax=axis,
        )
        axis.set(title=f"{model_name} — Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        figure.tight_layout()
        figure.savefig(ASSETS_DIR / f"confusion_matrix_{_model_slug(model_name)}.png", dpi=150)
        plt.close(figure)

    chart_frame = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision_weighted"],
                "Recall": result["recall_weighted"],
                "F1": result["f1_weighted"],
            }
            for name, result in results.items()
        ]
    ).set_index("Model")
    figure, axis = plt.subplots(figsize=(9, 5))
    chart_frame.plot(kind="bar", ax=axis, ylim=(0, 1), color=["#3478f6", "#39a77a", "#f0a24b", "#a46be0"])
    axis.set(title="Model Performance Comparison", ylabel="Score", xlabel="")
    axis.tick_params(axis="x", rotation=15)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(ASSETS_DIR / "model_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    axis.bar(["FAKE", "REAL"], [int(counts[0]), int(counts[1])], color=["#e45c63", "#36a77b"])
    axis.set(title="Dataset Class Distribution", ylabel="Number of articles")
    figure.tight_layout()
    figure.savefig(ASSETS_DIR / "class_distribution.png", dpi=150)
    plt.close(figure)


def train_model() -> dict[str, Any]:
    """Train three classifiers, compare their metrics, and save all artifacts."""
    ensure_project_dirs()
    dataset, demo_mode = load_dataset()
    dataset = dataset.copy()
    dataset["cleaned_content"] = dataset["content"].map(clean_text)
    dataset = dataset[dataset["cleaned_content"].str.strip().ne("")]
    if dataset["label"].nunique() != 2 or dataset["label"].value_counts().min() < 5:
        raise ValueError(
            "Training requires at least five usable articles in each class after text cleaning."
        )

    texts = dataset["cleaned_content"]
    labels = dataset["label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    vectorizer = _make_vectorizer()
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)
    estimators = _make_estimators()
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(x_train_vectors, y_train)
            predictions = model.predict(x_test_vectors)
            cv_pipeline = make_pipeline(_make_vectorizer(), clone(estimator))
            cv_scores = cross_val_score(
                cv_pipeline,
                x_train,
                y_train,
                cv=cv_splitter,
                scoring="accuracy",
                n_jobs=1,
            )
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        results[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_weighted": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "precision_real": float(
                precision_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "recall_real": float(
                recall_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "f1_binary_real": float(
                f1_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=_LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        }
        fitted_models[model_name] = model

    best_name = max(results, key=lambda name: results[name]["f1_weighted"])
    best_model = fitted_models[best_name]
    model_files: dict[str, str] = {}
    for model_name, model in fitted_models.items():
        filename = f"model_{_model_slug(model_name)}.joblib"
        joblib.dump(model, MODELS_DIR / filename)
        model_files[model_name] = filename
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    _save_plots(results, labels)
    metrics: dict[str, Any] = {
        "best_model": best_name,
        "demo_mode": demo_mode,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "feature_count": int(len(vectorizer.get_feature_names_out())),
        "model_files": model_files,
        "models": results,
        "top_indicative_words": _top_indicative_words(best_model, vectorizer),
        "plots": {
            "model_comparison": "model_comparison.png",
            "class_distribution": "class_distribution.png",
            "confusion_matrices": {
                name: f"confusion_matrix_{_model_slug(name)}.png" for name in results
            },
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision (weighted)": result["precision_weighted"],
                "Recall (weighted)": result["recall_weighted"],
                "F1 (weighted)": result["f1_weighted"],
                "F1 (REAL)": result["f1_binary_real"],
                "5-fold CV accuracy": result["cv_accuracy_mean"],
            }
            for name, result in results.items()
        ]
    ).sort_values("F1 (weighted)", ascending=False)
    print("\nModel comparison (stratified 80/20 holdout):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model: {best_name}")
    if demo_mode:
        print("WARNING: This model was trained on the built-in demo dataset, not the Kaggle dataset.")
    print(f"Saved metrics to {METRICS_PATH}")
    return metrics


def main() -> None:
    """Run training from the command line."""
    train_model()


if __name__ == "__main__":
    main()
