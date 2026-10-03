"""Streamlit interface for the fake news detection system."""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ASSETS_DIR, METRICS_PATH
from src.predict import (
    ArtifactBundle,
    get_influential_features,
    load_artifacts,
    predict_news,
)
from src.preprocessing import get_processing_steps
from src.train import train_model

st.set_page_config(
    layout="wide",
    page_icon="📰",
    page_title="Fake News Detector",
)

_REAL_SAMPLE = (
    "City council approves a flood protection plan after engineers reviewed "
    "recent risks and published their findings."
)
_FAKE_SAMPLE = (
    "Secret miracle drink cures every illness overnight, anonymous post promises "
    "guaranteed results without any medical evidence."
)
_DISCLAIMER = (
    "⚠️ The confidence score represents the model's prediction confidence, not proof "
    "that the news is factually true or false. Always verify with trusted sources."
)


@st.cache_resource
def cached_artifacts(model_name: str | None) -> ArtifactBundle:
    """Cache selected model artifacts for this Streamlit process."""
    return load_artifacts(model_name)


def _inject_styles() -> None:
    """Apply responsive styles for the page, cards, and results."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px;}
        .hero {padding: 2rem 2.2rem; border-radius: 20px; color: white;
          background: linear-gradient(120deg, #173c78 0%, #3868c5 54%, #7654bb 100%);
          box-shadow: 0 12px 30px rgba(36, 70, 130, .20); margin-bottom: 1.5rem;}
        .hero h1 {margin: 0; font-size: clamp(1.8rem, 3vw, 2.6rem); color: white;}
        .hero p {margin: .55rem 0 0; opacity: .92; font-size: 1.05rem;}
        .result-card {padding: 1.35rem 1.6rem; border-radius: 16px; margin: .7rem 0 1rem;
          box-shadow: 0 6px 20px rgba(20, 30, 50, .10); border: 1px solid transparent;}
        .result-real {background: linear-gradient(120deg, rgba(28, 150, 95, .15), rgba(28, 150, 95, .04));
          border-color: rgba(28, 150, 95, .35);}
        .result-fake {background: linear-gradient(120deg, rgba(220, 68, 78, .16), rgba(220, 68, 78, .04));
          border-color: rgba(220, 68, 78, .35);}
        .result-card h2 {margin: 0 0 .35rem; font-size: 1.65rem;}
        .result-card p {margin: .2rem 0; font-size: 1.05rem;}
        .metric-card {border-radius: 14px; padding: 1rem 1.1rem; background: rgba(120, 135, 160, .10);
          border: 1px solid rgba(120, 135, 160, .16); min-height: 90px;}
        .metric-label {font-size: .9rem; opacity: .78; margin-bottom: .25rem;}
        .metric-value {font-size: 1.6rem; font-weight: 700;}
        .token-highlight {background: #ffe08a; color: #27231b; padding: 1px 3px; border-radius: 4px;}
        .processed-text {white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7;}
        .small-muted {opacity: .72; font-size: .9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: float) -> None:
    """Render a compact styled model metric."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{value:.1%}</div></div>',
        unsafe_allow_html=True,
    )


def _clear_model_results() -> None:
    """Clear predictions when the serving model is changed."""
    st.session_state["latest_prediction"] = None
    st.session_state["batch_results"] = None


def _highlight_features(text: str, features: list[dict[str, Any]]) -> str:
    """Escape input and wrap model-influential text spans with a highlight."""
    words = [str(feature["word"]) for feature in features if feature.get("word")]
    if not words:
        return html.escape(text)
    alternatives = "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))
    matcher = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE)
    pieces: list[str] = []
    last_index = 0
    for match in matcher.finditer(text):
        pieces.append(html.escape(text[last_index : match.start()]))
        pieces.append(
            f'<span class="token-highlight">{html.escape(match.group(0))}</span>'
        )
        last_index = match.end()
    pieces.append(html.escape(text[last_index:]))
    return "".join(pieces)


def _render_detection(
    input_text: str, result: dict[str, Any], bundle: ArtifactBundle
) -> None:
    """Render the prediction, confidence, processing steps, and explanations."""
    model, vectorizer, _ = bundle
    label = result["label"]
    card_class = "result-real" if label == "REAL" else "result-fake"
    st.markdown(
        f'<div class="result-card {card_class}"><h2>Prediction: {label} NEWS</h2>'
        f'<p>Confidence: {result["confidence"]:.1f}%</p></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, max(0.0, result["confidence"] / 100.0)))
    probability_frame = pd.DataFrame(
        {"Probability": [result["prob_fake"], result["prob_real"]]},
        index=["FAKE", "REAL"],
    )
    st.bar_chart(probability_frame, y="Probability", color="#5278d7", height=190)
    st.caption(_DISCLAIMER)
    word_count = len(re.findall(r"\b[\w'-]+\b", input_text))
    st.caption(f"Input length: {word_count:,} words · {len(input_text):,} characters")
    if result["confidence"] < 60:
        st.warning("Low confidence — treat this result with extra caution.")

    steps = get_processing_steps(input_text)
    with st.expander("How the text was processed"):
        stages = (
            ("Original", steps["original"]),
            ("Tokens", " · ".join(steps["tokens"])),
            ("Stop-words removed", " · ".join(steps["tokens_without_stopwords"])),
            ("Final tokens", " · ".join(steps["final_tokens"])),
        )
        for heading, value in stages:
            st.markdown(f"**{heading}**")
            st.markdown(
                f'<div class="processed-text">{html.escape(value) or "(empty)"}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Top words influencing this prediction"):
        influential = get_influential_features(input_text, model, vectorizer)
        if influential:
            highlighted = _highlight_features(input_text, influential)
            st.markdown(
                f'<div class="processed-text">{highlighted}</div>',
                unsafe_allow_html=True,
            )
            st.caption("The strongest active model features are highlighted.")
            feature_frame = pd.DataFrame(influential)
            feature_frame["direction"] = feature_frame["impact"].map(
                lambda score: "toward REAL" if score > 0 else "toward FAKE"
            )
            st.dataframe(
                feature_frame[["word", "direction", "impact"]].rename(
                    columns={"word": "Feature", "direction": "Model direction", "impact": "Contribution"}
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("This selected model does not expose per-word feature weights.")


def _render_detect_tab(bundle: ArtifactBundle, model_name: str) -> None:
    """Render interactive single-article detection."""
    st.subheader("Analyze a headline or article")
    clear_column, _ = st.columns([1, 5])
    if clear_column.button("Clear text", use_container_width=True):
        st.session_state["news_input"] = ""
        st.session_state["latest_prediction"] = None
    input_text = st.text_area(
        "Paste news text",
        key="news_input",
        height=230,
        placeholder="Paste a headline or article here (at least five words)...",
        help="English-language text works best. Publisher and date fields are not used.",
    )
    if st.button("Analyze article", type="primary", use_container_width=True):
        try:
            with st.spinner("Cleaning text and analyzing with the selected model..."):
                st.session_state["latest_prediction"] = predict_news(
                    input_text, model_name=model_name, artifacts=bundle
                )
            st.session_state["prediction_input"] = input_text
        except ValueError as error:
            st.warning(str(error))
            st.session_state["latest_prediction"] = None
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")
            st.session_state["latest_prediction"] = None

    result = st.session_state.get("latest_prediction")
    if result and st.session_state.get("prediction_input") == input_text:
        _render_detection(
            st.session_state.get("prediction_input", input_text), result, bundle
        )
    elif result:
        st.info("The text has changed since the last analysis. Analyze it again to refresh the result.")


def _render_performance_tab(metrics: dict[str, Any]) -> None:
    """Show model comparisons, evaluation plots, and retraining controls."""
    st.subheader("Model performance")
    best_name = metrics["best_model"]
    best_metrics = metrics["models"][best_name]
    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ("Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"),
        ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"),
    ):
        with column:
            _metric_card(label, best_metrics[key])
    st.caption(f"Best model selected by weighted F1: **{best_name}**")

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
            for name, result in metrics["models"].items()
        ]
    ).sort_values("F1 (weighted)", ascending=False)
    st.dataframe(
        comparison.style.format(
            {column: "{:.3f}" for column in comparison.columns if column != "Model"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    comparison_plot = ASSETS_DIR / metrics["plots"]["model_comparison"]
    if comparison_plot.is_file():
        st.image(str(comparison_plot), caption="Accuracy, precision, recall, and F1 comparison")

    plot_columns = st.columns(3)
    for column, (model_name, filename) in zip(
        plot_columns, metrics["plots"]["confusion_matrices"].items()
    ):
        image_path = ASSETS_DIR / filename
        with column:
            if image_path.is_file():
                st.image(str(image_path), caption=f"{model_name} confusion matrix")
    distribution_plot = ASSETS_DIR / metrics["plots"]["class_distribution"]
    if distribution_plot.is_file():
        st.image(str(distribution_plot), caption="Training dataset class distribution", width=500)

    st.divider()
    if st.button("Retrain model", type="primary"):
        try:
            with st.spinner("Training and evaluating all three models..."):
                train_model()
            load_artifacts.cache_clear()
            cached_artifacts.clear()
            st.success("Training complete. Model artifacts and plots have been refreshed.")
            st.rerun()
        except Exception as error:
            st.error(f"Retraining failed: {error}")


def _render_batch_tab(bundle: ArtifactBundle, model_name: str) -> None:
    """Run selected-model predictions over a user-uploaded CSV."""
    st.subheader("Classify a CSV of articles")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV with a column containing headlines or article text.")
        return
    try:
        frame = pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="replace")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return
    if frame.empty or len(frame.columns) == 0:
        st.warning("The uploaded CSV has no rows or columns.")
        return
    text_column = st.selectbox("Text column", options=list(frame.columns))
    if st.button("Predict all rows", type="primary"):
        output = frame.copy()
        labels: list[str] = []
        confidences: list[float | None] = []
        errors: list[str | None] = []
        try:
            with st.spinner(f"Analyzing {len(frame):,} rows..."):
                for value in frame[text_column].fillna("").astype(str):
                    try:
                        prediction = predict_news(
                            value, model_name=model_name, artifacts=bundle
                        )
                        labels.append(f"{prediction['label']}")
                        confidences.append(round(prediction["confidence"], 2))
                        errors.append(None)
                    except ValueError as error:
                        labels.append("UNCLASSIFIED")
                        confidences.append(None)
                        errors.append(str(error))
            output["label"] = labels
            output["confidence"] = confidences
            output["prediction_error"] = errors
            st.session_state["batch_results"] = output
        except Exception as error:
            st.error(f"Batch prediction could not be completed: {error}")
            return

    output = st.session_state.get("batch_results")
    if output is not None:
        st.dataframe(output, hide_index=True, use_container_width=True)
        counts = output["label"].value_counts().reindex(["REAL", "FAKE"], fill_value=0)
        pie_frame = pd.DataFrame({"Label": counts.index, "Articles": counts.values})
        if int(counts.sum()) > 0:
            st.plotly_chart(
                px.pie(pie_frame, names="Label", values="Articles", color="Label"),
                use_container_width=True,
            )
        st.download_button(
            "Download results as CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="fake_news_predictions.csv",
            mime="text/csv",
        )


def _render_about_tab() -> None:
    """Explain the model workflow, intended use, and limitations."""
    st.subheader("About this project")
    st.markdown(
        """
        This educational NLP application estimates whether a supplied article resembles
        examples labeled **FAKE** or **REAL** in its training data. It is a text classifier,
        not a verification service.

        **Pipeline:** publisher/dateline cleanup and shared text preprocessing → TF-IDF
        unigram/bigram features → one of three trained classifiers → class probabilities
        and confidence display.

        **Objectives:** demonstrate dataset preparation, leakage reduction, reproducible
        model comparison, calibrated classification, and an interactive review workflow.

        **Technology:** Python, Pandas, NumPy, NLTK, scikit-learn, joblib, Matplotlib,
        Seaborn, Plotly, and Streamlit.

        **Limitations:** predictions can reflect dataset bias and may fail on new events,
        satire, opinion, or unfamiliar writing styles. A high score is not proof that a
        claim is true or false. The supplied dataset is English-language and results
        should be checked against trusted, independent sources.
        """
    )


def main() -> None:
    """Build the application page and handle artifact-loading errors."""
    _inject_styles()
    st.markdown(
        '<div class="hero"><h1>📰 Fake News Detection System</h1>'
        '<p>Explore how language patterns in a news story compare with labeled examples.</p></div>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Loading the selected model..."):
            default_bundle = cached_artifacts(None)
        metrics = default_bundle[2]
    except Exception as error:
        st.error(f"The model could not be loaded or trained: {error}")
        st.info("Check that dependencies are installed and the data directory is writable.")
        return

    if metrics.get("demo_mode"):
        st.warning(
            "Demo mode: the active model was trained on a tiny built-in sample, "
            "so its predictions are for demonstration only. Add the Kaggle CSV files and retrain."
        )

    with st.sidebar:
        st.header("Model")
        model_names = list(metrics.get("model_files", {})) or [metrics["best_model"]]
        if "selected_model" not in st.session_state or st.session_state["selected_model"] not in model_names:
            st.session_state["selected_model"] = metrics["best_model"]
        selected_model = st.selectbox(
            "Serving model",
            options=model_names,
            key="selected_model",
            on_change=_clear_model_results,
            help="All models were evaluated on the same stratified holdout set.",
        )
        st.caption(f"Active model: **{selected_model}**")
        best = metrics["models"][metrics["best_model"]]
        st.markdown("**Best-model holdout statistics**")
        st.metric("Accuracy", f"{best['accuracy']:.1%}")
        st.metric("Weighted F1", f"{best['f1_weighted']:.1%}")
        st.markdown("---")
        st.markdown("**Load a sample**")
        if st.button("Load real-style example", use_container_width=True):
            st.session_state["news_input"] = _REAL_SAMPLE
        if st.button("Load fake-style example", use_container_width=True):
            st.session_state["news_input"] = _FAKE_SAMPLE
        st.markdown("---")
        st.markdown(
            "**About**\n\nAn educational text classifier. Results are model estimates, "
            "not fact checks."
        )
        if METRICS_PATH.is_file():
            st.caption(f"Features used by the best model: {metrics.get('feature_count', 0):,}")

    try:
        bundle = (
            default_bundle
            if selected_model == metrics["best_model"]
            else cached_artifacts(selected_model)
        )
    except Exception as error:
        st.error(f"The selected model could not be loaded: {error}")
        return

    detect_tab, performance_tab, batch_tab, about_tab = st.tabs(
        ["🔍 Detect", "📊 Model Performance", "📁 Batch Prediction", "ℹ️ About"]
    )
    with detect_tab:
        _render_detect_tab(bundle, selected_model)
    with performance_tab:
        _render_performance_tab(metrics)
    with batch_tab:
        _render_batch_tab(bundle, selected_model)
    with about_tab:
        _render_about_tab()


if __name__ == "__main__":
    main()
