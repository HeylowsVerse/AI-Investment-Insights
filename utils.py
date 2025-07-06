import json
import logging
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def load_json(file) -> List[dict]:
    """Load JSON content from uploaded file."""
    if hasattr(file, "read"):
        data = json.load(file)
    else:
        with open(file) as f:
            data = json.load(f)
    return data


def parse_uploaded_files(uploaded_files: List) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Parse uploaded JSON files and organize them by company and type."""
    companies: Dict[str, Dict[str, pd.DataFrame]] = {}
    for upl in uploaded_files:
        name = Path(upl.name).stem
        parts = name.split("_")
        company = parts[0]
        ftype = "unknown"
        if "stock" in name.lower():
            ftype = "stock"
        elif "news" in name.lower():
            ftype = "news"
        elif "filing" in name.lower() or "sec" in name.lower():
            ftype = "filings"
        data = load_json(upl)
        # Convert JSON to DataFrame. If the JSON file represents a single
        # record (i.e. a dictionary of scalar values), wrap it in a list so
        # pandas does not raise a ValueError about missing index.
        if isinstance(data, dict) and not any(isinstance(v, (list, tuple, dict)) for v in data.values()):
            data = [data]
        df = pd.DataFrame(data)
        if company not in companies:
            companies[company] = {}
        companies[company][ftype] = df
    return companies


def compute_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, Bollinger Bands, and RSI to stock dataframe."""
    df = df.copy()
    # Normalize potential date column names
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp", "time"}:
            date_col = c
            break
    if not date_col:
        raise KeyError("date")

    df["date"] = pd.to_datetime(df[date_col])
    df.sort_values("date", inplace=True)

    # Identify the column containing closing prices
    close_col = None
    for c in df.columns:
        if c.lower() == "close":
            close_col = c
            break
    if not close_col:
        for c in df.columns:
            if "close" in c.lower():
                close_col = c
                break
    if not close_col:
        raise KeyError("close")

    if close_col != "close":
        df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    else:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df["return"] = df["close"].pct_change()
    for window in [20, 50, 200]:
        df[f"ma{window}"] = df["close"].rolling(window).mean()
    # Bollinger Bands
    df["std20"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * df["std20"]
    df["bb_lower"] = df["ma20"] - 2 * df["std20"]
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def analyze_sentiment(text: str) -> float:
    """Return sentiment polarity using TextBlob."""
    return TextBlob(text).sentiment.polarity


def extract_entities(text: str) -> List[str]:
    if not nlp:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def analyze_text_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    if text_col not in df.columns:
        possible = [
            col
            for col in df.columns
            if col.lower() in {
                "text",
                "content",
                "body",
                "article",
                "summary",
                "description",
                "headline",
            }
        ]
        if possible:
            text_col = possible[0]
        else:
            # Fall back to first string/object column if available
            object_cols = df.select_dtypes(include="object").columns
            if len(object_cols) > 0:
                text_col = object_cols[0]
            else:
                raise KeyError(text_col)

    df["sentiment"] = df[text_col].astype(str).apply(analyze_sentiment)
    df["entities"] = df[text_col].astype(str).apply(extract_entities)
    # Normalize potential date column names
    date_col = None
    if "date" in df.columns:
        date_col = "date"
    else:
        for c in df.columns:
            if c.lower() in {"datetime", "timestamp", "time", "published", "published_at"}:
                date_col = c
                break

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df

def company_summary(company: str, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return summary metrics for a company."""
    result = {"company": company}
    if "stock" in data:
        result["return_%"] = data["stock"]["return"].sum() * 100
        result["volatility"] = data["stock"]["return"].std() * np.sqrt(252)
    if "news" in data:
        result["avg_news_sentiment"] = data["news"]["sentiment"].mean()
    if "filings" in data:
        result["avg_filing_sentiment"] = data["filings"]["sentiment"].mean()
    return pd.DataFrame([result])


def news_analyze(json_file: Path, top_n: int = 20) -> Dict[str, float]:
    """Return a keyword density tag cloud from a JSON list of article URLs."""
    urls = load_json(json_file)
    if not isinstance(urls, list):
        logging.error("News JSON must contain a list of URLs")
        return {}

    aggregated_counts: Counter = Counter()
    total_words = 0

    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            body_text = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
        except Exception as e:  # pragma: no cover - network calls
            logging.error("Failed to fetch %s: %s", url, e)
            continue

        text = body_text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = [t for t in text.split() if t not in STOP_WORDS]
        if nlp:
            doc = nlp(" ".join(tokens))
            tokens = [t.lemma_ for t in doc if not t.is_space]

        aggregated_counts.update(tokens)
        total_words += len(tokens)

    if total_words == 0:
        return {}

    density = {word: count / total_words for word, count in aggregated_counts.items()}
    top = dict(sorted(density.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
    return top
