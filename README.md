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

