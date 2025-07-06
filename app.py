import io
import json
from typing import Dict

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    aggregate_sentiment,
    analyze_text_df,
    company_summary,
    compute_stock_indicators,
    parse_uploaded_files,
)
from scrape_data import SEMICONDUCTOR_COMPANIES

st.set_page_config(page_title="Investment Insights", layout="wide")

st.title("Multi-Company Investment Insight App")

# ---------- Data Extraction Section ----------
st.subheader("Data Extraction")
with st.form("scrape_form"):
    company_options = {f"{name} ({ticker})": ticker for ticker, name in SEMICONDUCTOR_COMPANIES.items()}
    tickers_input = st.multiselect(
        "Select Company", list(company_options.keys()), help="Up to 3 companies", max_selections=3
    )

    day_map = {"30D": 30, "60D": 60, "90D": 90, "120D": 120}
    days_label = st.selectbox("Days of historical prices", list(day_map.keys()), index=2)

    scrape_button = st.form_submit_button("Start")

if scrape_button:
    tickers = [company_options[sel] for sel in tickers_input]
    if len(tickers) > 3:
        st.error("Please select no more than 3 companies.")
    else:
        from scrape_data import main as scrape_main

        days_input = day_map[days_label]
        for tkr in tickers:
            with st.spinner(f"Scraping data for {tkr}..."):
                try:
                    result = scrape_main(tkr, int(days_input))
                    st.success(f"Scraped data for {tkr}. Download files below.")
                    stock_data = json.dumps(result["stock"], indent=4).encode("utf-8")
                    st.download_button(
                        f"Download Stock Data ({tkr})",
                        stock_data,
                        file_name=f"{tkr}_stock.json",
                    )
                    if result.get("filings"):
                        filing_data = json.dumps(result["filings"], indent=4).encode("utf-8")
                        st.download_button(
                            f"Download Filing ({tkr})",
                            filing_data,
                            file_name=f"{tkr}_filing_latest.json",
                        )
                    news_data = json.dumps(result["news"], indent=4).encode("utf-8")
                    st.download_button(
                        f"Download News ({tkr})",
                        news_data,
                        file_name=f"{tkr}_news_{result['timestamp']}.json",
                    )
                except Exception as e:  # pragma: no cover - manual operation
                    st.error(f"Failed to scrape data for {tkr}: {e}")

uploaded = st.file_uploader(
    "Upload JSON files (max 3 companies)",
    accept_multiple_files=True,
    type="json",
)

if uploaded:
    companies = parse_uploaded_files(uploaded)

    summaries = []
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
                rsi_chart = alt.Chart(data["stock"]).mark_line().encode(
                    x="date:T", y="rsi:Q"
                ).properties(title="RSI")
                st.altair_chart(rsi_chart, use_container_width=True)
                csv = data["stock"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Stock Data", data=csv, file_name=f"{company}_stock.csv"
                )
            if "news" in data:
                data["news"] = analyze_text_df(data["news"])
                trend = aggregate_sentiment(data["news"])
                if not trend.empty and "date" in trend.columns:
                    fig = px.line(trend, x="date", y="sentiment", title="News Sentiment")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("News sentiment trend unavailable: no dates provided.")
            if "filings" in data:
                data["filings"] = analyze_text_df(data["filings"])
                word_freq = pd.DataFrame(data["filings"]["keywords"].tolist()).sum()
                word_chart = alt.Chart(
                    word_freq.reset_index().rename({"index": "keyword", 0: "count"}, axis=1)
                ).mark_bar().encode(x="keyword", y="count")
                st.altair_chart(word_chart, use_container_width=True)
            summaries.append(company_summary(company, data))

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        st.subheader("Company Comparison")
        st.dataframe(summary_df)
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Summary", data=csv, file_name="summary.csv")
else:
    st.info(
        "Use the form above to scrape new data or upload existing JSON files "
        "generated by `scrape_data.py`."
    )
