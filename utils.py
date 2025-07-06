import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from textblob import TextBlob
import spacy

nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

KEYWORDS = ["AI", "M&A", "earnings", "guidance"]


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


def keyword_frequency(text: str, keywords=KEYWORDS) -> Dict[str, int]:
    text_lower = text.lower()
    return {k: text_lower.count(k.lower()) for k in keywords}


def extract_entities(text: str) -> List[str]:
    if not nlp:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def analyze_text_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df[text_col].apply(analyze_sentiment)
    df["keywords"] = df[text_col].apply(keyword_frequency)
    df["entities"] = df[text_col].apply(extract_entities)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df.groupby(pd.Grouper(key="date", freq="D"))["sentiment"].mean().reset_index()
    else:
        return pd.DataFrame()


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
