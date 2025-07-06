import io
import json
import zipfile
from typing import Dict

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from utils import (
    analyze_text_df,
    company_summary,
    compute_stock_indicators,
    parse_uploaded_files,
    news_analyze,
)
from pathlib import Path
from scrape_data import SEMICONDUCTOR_COMPANIES


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return (
            df.groupby(pd.Grouper(key="date", freq="D"))["sentiment"].mean().reset_index()
        )
    return pd.DataFrame()

st.set_page_config(page_title="Investment Insights", layout="wide")

st.title("Multi-Company Investment Insight App")

# ---------- Data Extraction Section ----------
st.subheader("Data Extraction")
with st.form("scrape_form"):
    company_options = {
        f"{name} ({ticker})": ticker for ticker, name in SEMICONDUCTOR_COMPANIES.items()
    }
    ticker_choice = st.selectbox(
        "Select Company",
        list(company_options.keys()),
        help="Select one company",
    )

    day_map = {"30D": 30, "60D": 60, "90D": 90, "120D": 120}
    days_label = st.selectbox(
        "Days of historical prices", list(day_map.keys()), index=2
    )

    scrape_button = st.form_submit_button("Start")

if scrape_button:
    ticker = company_options[ticker_choice]
    from scrape_data import main as scrape_main

    days_input = day_map[days_label]
    with st.spinner(f"Scraping data for {ticker}..."):
        try:
            result = scrape_main(ticker, int(days_input))
            st.success(f"Scraped data for {ticker}. Download file below.")
            stock_data = json.dumps(result["stock"], indent=4).encode("utf-8")
            news_data = json.dumps(result["news"], indent=4).encode("utf-8")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr(f"{ticker}_stock.json", stock_data)
                if result.get("filings"):
                    filing_data = json.dumps(
                        result["filings"], indent=4
                    ).encode("utf-8")
                    zf.writestr(f"{ticker}_filing_latest.json", filing_data)
                zf.writestr(f"{ticker}_news_{result['timestamp']}.json", news_data)
            zip_buffer.seek(0)
            st.download_button(
                f"Download {ticker} Data (zip)",
                zip_buffer,
                file_name=f"{ticker}_data_{result['timestamp']}.zip",
                mime="application/zip",
            )
        except Exception as e:  # pragma: no cover - manual operation
            st.error(f"Failed to scrape data for {ticker}: {e}")

uploaded = st.file_uploader(
    "Upload JSON files",
    accept_multiple_files=True,
    type="json",
)

if uploaded:
    news_clouds: Dict[str, Dict[str, float]] = {}
    for upl in uploaded:
        name = Path(upl.name).stem
        company = name.split("_")[0]
        try:
            data = json.load(upl)
        except Exception:
            upl.seek(0)
            continue
        upl.seek(0)
        if isinstance(data, list) and all(isinstance(u, str) for u in data):
            cloud = news_analyze(upl)
            if cloud:
                news_clouds[company] = cloud
        upl.seek(0)

    companies = parse_uploaded_files(uploaded)

    summaries = []
    sentiment_trends = []
    tabs = st.tabs(list(companies.keys()))
    for (company, data), tab in zip(companies.items(), tabs):
        with tab:
            st.header(company)
            if "stock" in data:
                data["stock"] = compute_stock_indicators(data["stock"])
                fig = px.line(
                    data["stock"],
                    x="date",
                    y=["close", "ma20", "ma50", "ma200"],
                    title=f"{company} Price & MAs",
                )
                st.plotly_chart(fig, use_container_width=True)
                rsi_chart = (
                    alt.Chart(data["stock"])
                    .mark_line()
                    .encode(x="date:T", y="rsi:Q")
                    .properties(title="RSI")
                )
                st.altair_chart(rsi_chart, use_container_width=True)
                csv = data["stock"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Stock Data", data=csv, file_name=f"{company}_stock.csv"
                )
            if "news" in data:
                data["news"] = analyze_text_df(data["news"])
                trend = aggregate_sentiment(data["news"])
                if not trend.empty and "date" in trend.columns:
                    trend["company"] = company
                    sentiment_trends.append(trend)
                if company in news_clouds:
                    st.subheader("News Sentiment Tag Cloud")
                    wc = WordCloud(width=800, height=400, background_color="white")
                    wc = wc.generate_from_frequencies(news_clouds[company])
                    st.image(wc.to_array(), use_container_width=True)
            if "filings" in data:
                data["filings"] = analyze_text_df(data["filings"])
                word_freq = (
                    pd.DataFrame(data["filings"]["keywords"].tolist())
                    .sum()
                    .astype(int)
                )
                word_freq = word_freq[word_freq > 0]
                if not word_freq.empty:
                    wc = WordCloud(width=800, height=400, background_color="white")
                    wc = wc.generate_from_frequencies(word_freq.to_dict())
                    st.image(wc.to_array(), use_container_width=True)
                else:
                    st.write("No keyword frequencies found in filings.")
            summaries.append(company_summary(company, data))

    if len(sentiment_trends) > 1:
        all_trends = pd.concat(sentiment_trends, ignore_index=True)
        st.subheader("News Sentiment Comparison")
        fig = px.line(
            all_trends,
            x="date",
            y="sentiment",
            color="company",
            title="News Sentiment by Company",
        )
        st.plotly_chart(fig, use_container_width=True)

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        st.subheader("Company Comparison")
        st.dataframe(summary_df)
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Summary", data=csv, file_name="summary.csv")
else:
    st.info(
        "Use the form above to scrape new data or upload existing JSON files."
    )
