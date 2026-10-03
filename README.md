# Fake News Detection System

An educational NLP application that classifies a news headline or article as **FAKE**
or **REAL**, displays calibrated class probabilities, and compares three classical
machine-learning models. The Streamlit interface also supports CSV batch predictions.
Predictions are model estimates, not independent fact checks.

## Project structure

```text
fake_news_detector/
├── data/                      # Place Fake.csv and True.csv here
├── models/                    # Trained model, vectorizer, and metrics (created automatically)
├── assets/                    # Evaluation charts (created automatically)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or later. From the `fake_news_detector` directory, create and
activate a virtual environment:

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NLTK attempts to download tokenizer, English stop-word, and WordNet resources on
first use. If those resources cannot be reached, tokenization and stop-word removal
have offline fallbacks and lemmatization safely leaves tokens unchanged.

## Dataset

Download the Kaggle [**Fake and Real News Dataset**](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(commonly published by Clément Bisaillon). Put the original files in the
project's `data/` folder with these exact names:

```text
fake_news_detector/
└── data/
    ├── Fake.csv
    └── True.csv
```

Both files should include `title` and `text` columns; `subject` and `date` are
not model features. Training combines title and article text, removes duplicates
and empty rows, strips Reuters datelines/source tags and image-credit boilerplate,
and labels fake articles `0` and real articles `1`.

If either CSV is missing, the app and training command use a balanced, tiny
built-in set of 20 hand-written examples. A visible warning identifies this as
**demo mode**; its evaluation scores are not representative of real-world
performance. Add the Kaggle files and choose **Retrain model** to train on them.

## Train and run

From the project directory:

```powershell
python -m src.train
streamlit run app.py
```

The first app launch automatically trains the models if artifacts do not yet
exist. Training writes `models/best_model.joblib`,
`models/tfidf_vectorizer.joblib`, individual classifier files, and
`models/metrics.json`; it also saves charts in `assets/`. The Streamlit app
provides single-text analysis, per-feature explanations for linear models,
model-performance visualizations, and CSV batch prediction.

## Sample output

```text
Prediction: REAL NEWS
Confidence: 96.1%
```

The probability chart shows both class probabilities. Confidence is the larger
class probability from the selected model. Treat uncertain or high-confidence
predictions as a starting point for review, not as a factual verdict.

## Evaluation metrics

Training uses a reproducible stratified 80/20 train/test split. TF-IDF features
are fitted using training data only. Logistic Regression, Multinomial Naive
Bayes, and a probability-calibrated Linear SVM are evaluated on the same holdout
set and with five-fold cross-validation on the training portion.

- **Accuracy:** fraction of holdout predictions that are correct.
- **Precision (weighted):** class-wise precision averaged by each class's support.
- **Recall (weighted):** class-wise recall averaged by each class's support.
- **F1 (weighted):** support-weighted harmonic mean of precision and recall;
  this selects the saved best model.
- **F1 (REAL):** binary F1 with REAL treated as the positive class.
- **Confusion matrix / classification report:** per-class error counts and
  precision, recall, and F1.
- **Five-fold CV accuracy:** mean accuracy across five training-set folds.

The Linear SVM uses `CalibratedClassifierCV` so the interface can report
probability estimates. Calibration does not make those scores proof of truth.

## Limitations

The model learns patterns and biases in its source dataset; source-tag cleanup
reduces one form of leakage but cannot remove all dataset artifacts. Language,
topics, publication styles, and class balance may differ in live news. It is
English-only, can misclassify satire/opinion and breaking news, and cannot
verify claims against evidence. Do not rely on it for consequential decisions.

## Future improvements

- Evaluate transformer models such as BERT on leakage-resistant splits.
- Add multilingual preprocessing and language-specific evaluation.
- Integrate trusted fact-check APIs and source citations.
- Add temporal and publisher-independent holdout evaluations.
