"""Dataset loading, source-tag cleanup, and built-in demo examples."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import FAKE_CSV_PATH, REAL_CSV_PATH

_SOURCE_PATTERNS = (
    re.compile(
        r"(?:\b[A-Z][A-Z .,'&-]{0,45}\s+)?\(\s*Reuters\s*\)\s*[-–—:]?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bReuters\b", flags=re.IGNORECASE),
    re.compile(r"(?:featured\s+image|image)\s+via\b[^\r\n]*", flags=re.IGNORECASE),
)

_DEMO_ROWS = [
    (1, "Local schools expand free breakfast program", "District leaders approved funding to provide free breakfast at every public elementary school beginning next fall."),
    (1, "City council approves new flood protection plan", "The plan adds drainage upgrades and river monitoring after engineers reviewed the area's recent flood risks."),
    (1, "Researchers report progress on battery recycling", "A university team says its pilot process recovered several battery materials, with results published in a peer-reviewed journal."),
    (1, "Hospital opens a new community health clinic", "The clinic will offer primary care appointments and vaccination services, according to the hospital's public announcement."),
    (1, "Rail operator announces weekend maintenance schedule", "Service will run less frequently on two lines while crews replace track components during the scheduled overnight work."),
    (1, "County publishes annual drinking water results", "The report lists measurements from community water tests and says all samples met current federal standards."),
    (1, "Public library adds evening hours at three branches", "The library system said the extended schedule follows a six-month pilot and will begin on Monday."),
    (1, "Scientists track seasonal changes in coastal birds", "Researchers counted nesting sites along the coast and said the survey will continue through the summer migration."),
    (1, "Small businesses receive grants for energy upgrades", "The city awarded grants to shops installing efficient lighting and heating equipment under its published program."),
    (1, "Transit agency releases audited passenger figures", "An independent audit found ridership increased on several routes compared with the same period last year."),
    (0, "Secret device makes anyone rich overnight", "An anonymous online post promises guaranteed profits and urges readers to send money immediately to claim access."),
    (0, "Doctors stunned by miracle drink that cures every illness", "A viral claim says one kitchen drink replaces all medicine, but offers no study, named expert, or clinical evidence."),
    (0, "Government hiding proof that the moon is artificial", "A sensational message claims unnamed insiders revealed a secret without documents or verifiable sources."),
    (0, "Celebrity endorses impossible weight loss trick", "The article uses a fabricated quote and demands payment for a product supposedly guaranteed to work for everyone."),
    (0, "New law will ban all internet use next week", "A widely shared post warns of an imminent nationwide shutdown but links to no official legislation or announcement."),
    (0, "Scientists confirm people can live forever with this herb", "The post claims immortality is proven while citing no named researchers, journal, or reproducible findings."),
    (0, "Election results changed by hidden machines, viral post says", "The allegation repeats an unsupported conspiracy theory and provides no records or independently verified evidence."),
    (0, "Banks will erase every savings account tomorrow", "An anonymous warning tells readers to withdraw cash at once and includes no statement from a bank or regulator."),
    (0, "One strange signal proves aliens control the weather", "The story draws a sweeping conclusion from an unexplained signal without evidence connecting it to weather."),
    (0, "Free cash giveaway requires sharing private account details", "A fake promotion promises instant money and asks users to disclose passwords and financial information."),
]


def clean_source_tags(text: str) -> str:
    """Remove publisher identifiers and image-credit boilerplate."""
    cleaned = str(text)
    for pattern in _SOURCE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _read_news_file(path: Path, label: int) -> pd.DataFrame:
    """Read and validate a labeled Kaggle-format CSV file."""
    frame = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    required = {"title", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}.")
    title = frame["title"].fillna("").astype(str)
    body = frame["text"].fillna("").astype(str)
    content = (title + " " + body).map(clean_source_tags)
    return pd.DataFrame({"content": content, "label": label})


def load_dataset() -> tuple[pd.DataFrame, bool]:
    """Load Kaggle CSV files or return the balanced built-in demo dataset."""
    if not FAKE_CSV_PATH.is_file() or not REAL_CSV_PATH.is_file():
        print(
            "Dataset files not found. Download the Kaggle 'Fake and Real News Dataset' "
            "and place Fake.csv and True.csv in the data/ directory. "
            "Using the built-in demo examples for this run."
        )
        return pd.DataFrame(_DEMO_ROWS, columns=["label", "title", "text"]).assign(
            content=lambda frame: (
                frame["title"] + " " + frame["text"]
            ).map(clean_source_tags)
        )[["content", "label"]], True

    fake_frame = _read_news_file(FAKE_CSV_PATH, 0)
    real_frame = _read_news_file(REAL_CSV_PATH, 1)
    dataset = pd.concat([fake_frame, real_frame], ignore_index=True)
    dataset["content"] = dataset["content"].fillna("").astype(str).str.strip()
    dataset = dataset[dataset["content"].ne("")]
    dataset = dataset.drop_duplicates(subset="content").reset_index(drop=True)
    if dataset.empty or dataset["label"].nunique() != 2:
        raise ValueError("The dataset must contain non-empty examples for both Fake.csv and True.csv.")
    return dataset[["content", "label"]], False
