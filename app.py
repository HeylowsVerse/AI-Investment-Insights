import io
import json
import zipfile
from typing import Dict

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

import utils
from msi_forecast import (
    generate_sample_msi,
    interpolate_monthly,
    sarimax_forecast,
    simulate_pmi,
    regression_predict,
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
            cloud = utils.news_analyze(upl)
            if cloud:
                news_clouds[company] = cloud
        upl.seek(0)

    companies = utils.parse_uploaded_files(uploaded)

    summaries = []
    sentiment_trends = []
    tabs = st.tabs(list(companies.keys()))
    for (company, data), tab in zip(companies.items(), tabs):
        with tab:
            st.header(company)
            if "stock" in data:
                data["stock"] = utils.compute_stock_indicators(data["stock"])
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
                data["news"] = utils.analyze_text_df(data["news"])
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
                data["filings"] = utils.analyze_text_df(data["filings"])
            summaries.append(utils.company_summary(company, data))

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

st.header("MSI Forecast")
msi_q = generate_sample_msi()
msi_m = interpolate_monthly(msi_q)
forecast, ci, _ = sarimax_forecast(msi_m)
future_idx = ci.index
all_dates = msi_m.index.append(future_idx)
pmi = simulate_pmi(all_dates)
reg_pred, slope, intercept, rmse = regression_predict(
    msi_m, pmi.loc[msi_m.index], pmi.loc[future_idx]
)
reg_pred = pd.Series(reg_pred, index=future_idx, name="PMI Regression")

plot_df = (
    pd.concat(
        [msi_m.rename("Actual MSI"), forecast.rename("SARIMAX Forecast"), reg_pred],
        axis=1,
    )
    .reset_index()
    .rename(columns={"index": "date"})
)
ci_df = ci.reset_index().rename(columns={"index": "date"})

band = (
    alt.Chart(ci_df)
    .mark_area(opacity=0.3)
    .encode(x="date:T", y="lower msi:Q", y2="upper msi:Q")
)

lines = (
    alt.Chart(plot_df)
    .transform_fold(["Actual MSI", "SARIMAX Forecast", "PMI Regression"], as_=["type", "msi"])
    .mark_line()
    .encode(x="date:T", y="msi:Q", color="type:N")
)

vline = (
    alt.Chart(pd.DataFrame({"date": [msi_m.index[-1]]}))
    .mark_rule(color="gray", strokeDash=[4, 2])
    .encode(x="date:T")
)

st.altair_chart(band + lines + vline, use_container_width=True)
st.caption(
    f"Regression slope: {slope:.3f} intercept: {intercept:.3f} RMSE: {rmse:.3f}"
)
