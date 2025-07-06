# AI Investment Insights App

This repository contains a simple Streamlit application for analyzing investment data for a single company at a time. Users can upload JSON files containing stock price data, news articles, and SEC filings. The app applies basic quantitative finance and NLP techniques to generate insights and visualizations.

## Running the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m textblob.download_corpora
   python -m spacy download en_core_web_sm
   ```

2. Generate JSON files using the scraping script (or scrape directly in the app):
   ```bash
   python scrape_data.py TICKER --days 60
   ```

3. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

## Features
- Upload and assign JSON files to companies
- Stock price indicators (moving averages, Bollinger Bands, RSI)
- News and filing sentiment analysis with keyword cloud visualization
- Tag cloud generation from article URLs via `news_analyze`
- Cross-company news sentiment comparison charts and downloadable CSV summaries
- Scrape data directly in the app with download buttons


