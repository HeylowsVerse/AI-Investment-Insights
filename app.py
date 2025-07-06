import io
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

st.set_page_config(page_title="Investment Insights", layout="wide")

st.title("Multi-Company Investment Insight App")

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
                fig = px.line(trend, x="date", y="sentiment", title="News Sentiment")
                st.plotly_chart(fig, use_container_width=True)
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
    st.info("Upload company JSON files to begin.")
