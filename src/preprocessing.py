"""Text normalization shared by model training and serving."""

from __future__ import annotations

import re
from typing import Any

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

from src.config import USE_PORTER_STEMMER

_RESOURCE_PATHS = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
}
_STOP_WORDS: set[str] | None = None
_NLTK_RESOURCES_CHECKED = False
_LEMMATIZER = WordNetLemmatizer()
_STEMMER = PorterStemmer()
_FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}


def _prepare_nltk_resources() -> None:
    """Attempt to download NLTK resources without preventing offline use."""
    global _NLTK_RESOURCES_CHECKED
    if _NLTK_RESOURCES_CHECKED:
        return
    for resource_path, package in _RESOURCE_PATHS.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True, raise_on_error=False)
            except (OSError, ValueError):
                pass
    _NLTK_RESOURCES_CHECKED = True


def _get_stop_words() -> set[str]:
    """Return NLTK English stop words or a bundled offline fallback."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _prepare_nltk_resources()
        try:
            from nltk.corpus import stopwords

            _STOP_WORDS = set(stopwords.words("english"))
        except (LookupError, OSError):
            _STOP_WORDS = _FALLBACK_STOP_WORDS
    return _STOP_WORDS


def _tokenize(text: str) -> list[str]:
    """Tokenize text with NLTK and fall back to whitespace splitting."""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        _prepare_nltk_resources()
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            return text.split()


def _lemmatize(token: str) -> str:
    """Lemmatize or stem a token, tolerating a missing WordNet corpus."""
    if USE_PORTER_STEMMER:
        return _STEMMER.stem(token)
    try:
        return _LEMMATIZER.lemmatize(token)
    except LookupError:
        _prepare_nltk_resources()
        try:
            return _LEMMATIZER.lemmatize(token)
        except LookupError:
            return token


def get_processing_steps(text: str) -> dict[str, Any]:
    """Return the original text and its token-processing stages."""
    original = "" if text is None else str(text)
    lowered = original.lower()
    without_urls = re.sub(r"(?:https?://|www\.)\S+", " ", lowered)
    without_html = re.sub(r"<[^>]*>", " ", without_urls)
    without_emails = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", without_html)
    without_digits = re.sub(r"\d+", " ", without_emails)
    letters_only = re.sub(r"[^a-z\s]", " ", without_digits)
    tokens = _tokenize(re.sub(r"\s+", " ", letters_only).strip())
    tokens_without_stopwords = [
        token for token in tokens if token not in _get_stop_words()
    ]
    final_tokens = [
        normalized
        for token in tokens_without_stopwords
        if len(token) >= 3
        for normalized in [_lemmatize(token)]
        if len(normalized) >= 3
    ]
    return {
        "original": original,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_without_stopwords,
        "final_tokens": final_tokens,
        "cleaned_text": " ".join(final_tokens),
    }


def clean_text(text: str) -> str:
    """Normalize text consistently for both training and prediction."""
    return get_processing_steps(text)["cleaned_text"]
