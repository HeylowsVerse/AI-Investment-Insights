# AI Investment Insights App

This repository contains a simple Streamlit application for analyzing investment data for up to three companies. Users can upload JSON files containing stock price data, news articles, and SEC filings. The app applies basic quantitative finance and NLP techniques to generate insights and visualizations.

## Running the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m textblob.download_corpora
   python -m spacy download en_core_web_sm
   ```

2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

## Features
- Upload and assign JSON files to companies
- Stock price indicators (moving averages, Bollinger Bands, RSI)
- News and filing sentiment analysis with keyword frequency
- Cross-company comparison charts and downloadable CSV summaries

## Data Scraping

Use `scrape_data.py` to download stock prices, the latest SEC filing and
recent news for a single ticker. Example:

```bash
python scrape_data.py NVDA --days 60
```

The command saves JSON files such as `NVDA_stock.json`,
`NVDA_filing_latest.json` and `NVDA_news_YYYYMMDD_HHMM.json` in the current
directory.

