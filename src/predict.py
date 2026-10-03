"""Load trained artifacts and classify incoming news text."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.config import (
    BEST_MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    VECTORIZER_PATH,
    ensure_project_dirs,
)
from src.preprocessing import clean_text

ArtifactBundle = tuple[Any, Any, dict[str, Any]]


@lru_cache(maxsize=4)
def load_artifacts(model_name: str | None = None) -> ArtifactBundle:
    """Load the selected model, shared vectorizer, and evaluation metrics."""
    ensure_project_dirs()
    if not BEST_MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file() or not METRICS_PATH.is_file():
        from src.train import train_model

        train_model()
    if not METRICS_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "Model artifacts are not available. Add the Kaggle CSV files to data/ or check the training output."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected_name = model_name or metrics.get("best_model")
    if selected_name == metrics.get("best_model"):
        model_path = BEST_MODEL_PATH
    else:
        filename = metrics.get("model_files", {}).get(selected_name)
        if not filename:
            available = ", ".join(metrics.get("model_files", {})) or "the saved best model"
            raise ValueError(f"Unknown model '{selected_name}'. Available models: {available}.")
        model_path = MODELS_DIR / Path(filename).name
    if not model_path.is_file():
        raise FileNotFoundError(
            f"The saved model '{selected_name}' is missing. Use Retrain model to create it."
        )
    return joblib.load(model_path), joblib.load(VECTORIZER_PATH), metrics


def predict_news(
    text: str,
    model_name: str | None = None,
    artifacts: ArtifactBundle | None = None,
) -> dict[str, Any]:
    """Predict REAL/FAKE and return calibrated class probabilities."""
    if text is None or not str(text).strip():
        raise ValueError("Enter a headline or article before analyzing.")
    text = str(text)
    if len(re.findall(r"\b[\w'-]+\b", text)) < 5:
        raise ValueError("Please enter at least five words so the model has enough context.")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("No usable English words remain after text preprocessing. Please try different text.")
    model, vectorizer, _ = artifacts or load_artifacts(model_name)
    features = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(features)[0]
    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    fake_probability = class_probabilities.get(0, 0.0)
    real_probability = class_probabilities.get(1, 0.0)
    label = "REAL" if real_probability >= fake_probability else "FAKE"
    confidence = max(fake_probability, real_probability) * 100.0
    return {
        "label": label,
        "confidence": confidence,
        "prob_fake": fake_probability,
        "prob_real": real_probability,
        "cleaned_text": cleaned_text,
    }


def get_influential_features(
    text: str, model: Any, vectorizer: Any, limit: int = 10
) -> list[dict[str, float | str]]:
    """Rank present TF-IDF terms by their contribution to the model score."""
    weights: Any | None = None
    if hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    elif hasattr(model, "coef_"):
        weights = model.coef_[0]
    else:
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            estimator = getattr(
                calibrated[0],
                "estimator",
                getattr(calibrated[0], "base_estimator", None),
            )
            if estimator is not None and hasattr(estimator, "coef_"):
                weights = estimator.coef_[0]
    if weights is None:
        return []

    vector = vectorizer.transform([clean_text(text)]).tocsr()
    if vector.nnz == 0:
        return []
    contributions = vector.data * weights[vector.indices]
    ranked = sorted(
        zip(vector.indices, contributions),
        key=lambda pair: abs(float(pair[1])),
        reverse=True,
    )
    terms = vectorizer.get_feature_names_out()
    return [
        {"word": str(terms[index]), "impact": float(score)}
        for index, score in ranked[:limit]
    ]
