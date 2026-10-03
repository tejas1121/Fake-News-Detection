"""Shared paths and settings for the fake news detector."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
FAKE_CSV_PATH = DATA_DIR / "Fake.csv"
REAL_CSV_PATH = DATA_DIR / "True.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_SEED = 42
USE_PORTER_STEMMER = False
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.9


def ensure_project_dirs() -> None:
    """Create project directories needed for data, models, and plots."""
    for directory in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
