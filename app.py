import io
import json
import zipfile
from typing import Dict
from pathlib import Path
import os

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import InferenceClient

import utils
from msi_forecast import (
    generate_sample_msi,
    interpolate_monthly,
    safe_forecast,
    simulate_pmi,
    regression_predict,
)
from scrape_data import SEMICONDUCTOR_COMPANIES


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return (
            df.groupby(pd.Grouper(key="date", freq="D"))["sentiment"].mean().reset_index()
        )
    return pd.DataFrame()


def explain_chart_with_gemma(prompt: str, temperature=0.7, max_tokens=300) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating explanation: {e}"


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
                    filing_data = json.dumps(result["filings"], indent=4).encode("utf-8")
                    zf.writestr(f"{ticker}_filing_latest.json", filing_data)
                zf.writestr(f"{ticker}_news_{result['timestamp']}.json", news_data)
            zip_buffer.seek(0)
            st.download_button(
                f"Download {ticker} Data (zip)",
                zip_buffer,
                file_name=f"{ticker}_data_{result['timestamp']}.zip",
                mime="application/zip",
            )
        except Exception as e:
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

                chart_prompt = (
                    f"You are a financial analyst.\nThis chart shows the {company}'s stock price trends, including moving averages.\n"
                    f"Explain what this indicates about the company's recent performance and investor sentiment."
                )
                st.markdown("**📊 Chart Analysis**")
                st.write(explain_chart_with_gemma(chart_prompt))

                rsi_chart = (
                    alt.Chart(data["stock"])
                    .mark_line()
                    .encode(x="date:T", y="rsi:Q")
                    .properties(title="RSI")
                )
                st.altair_chart(rsi_chart, use_container_width=True)

                rsi_prompt = (
                    f"This RSI chart for {company} reflects momentum and potential overbought/oversold conditions.\n"
                    f"Write a brief summary interpreting current signals and implications."
                )
                st.markdown("**📈 RSI Analysis**")
                st.write(explain_chart_with_gemma(rsi_prompt))

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

        trend_prompt = (
            "This line chart compares news sentiment over time across multiple companies.\n"
            "Summarize the key sentiment trends and differences among the firms."
        )
        st.markdown("**🧠 Sentiment Trend Insight**")
        st.write(explain_chart_with_gemma(trend_prompt))

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        st.subheader("Company Comparison")
        st.dataframe(summary_df)
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Summary", data=csv, file_name="summary.csv")
else:
    st.info("Use the form above to scrape new data or upload existing JSON files.")

# ---------- MSI Forecast Section ----------
st.header("MSI Forecast")

# 1. Forecasting
msi_q = generate_sample_msi()
msi_m = interpolate_monthly(msi_q)
forecast, ci, _ = safe_forecast(msi_m)
future_idx = ci.index
all_dates = msi_m.index.append(future_idx)
pmi = simulate_pmi(all_dates)

# 2. Regression prediction
reg_pred, slope, intercept, rmse = regression_predict(
    msi_m, pmi.loc[msi_m.index], pmi.loc[future_idx]
)
reg_pred = pd.Series(reg_pred, index=future_idx, name="PMI Regression")

# 3. Combine into plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=msi_m.index, y=msi_m.values, mode="lines+markers", name="Actual MSI", yaxis="y1"))
fig.add_trace(go.Scatter(x=future_idx, y=forecast.values, mode="lines+markers", name="SARIMAX Forecast", line=dict(dash="dash"), yaxis="y1"))
fig.add_trace(go.Scatter(x=future_idx.tolist() + future_idx[::-1].tolist(), y=ci["upper msi"].tolist() + ci["lower msi"][::-1].tolist(), fill="toself", fillcolor="rgba(173,216,230,0.3)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=True, name="95% CI", yaxis="y1"))
fig.add_trace(go.Scatter(x=future_idx, y=reg_pred.values, mode="lines+markers", name="PMI Regression", line=dict(color="orange", dash="dot"), yaxis="y1"))
fig.add_trace(go.Scatter(x=all_dates, y=pmi.values, mode="lines", name="PMI", line=dict(color="gray"), yaxis="y2"))

fig.update_layout(
    title="MSI Forecast with PMI Overlay",
    xaxis=dict(title="Date"),
    yaxis=dict(title="MSI (Million Square Inches)", side="left"),
    yaxis2=dict(title="PMI", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Regression slope: {slope:.3f}, intercept: {intercept:.3f}, RMSE: {rmse:.2f}")

forecast_prompt = (
    f"You are a financial analyst.\nThe MSI forecast shows SARIMAX projections, confidence bands, and PMI overlay.\n"
    f"Explain how the forecast trend and PMI alignment reflect potential industry outlook."
)
st.markdown("**📉 Forecast Analysis**")
st.write(explain_chart_with_gemma(forecast_prompt))

# ---------- Gemma Setup ----------
model_id = "google/gemma-2b-it"
token = os.getenv("HF_TOKEN")
client = InferenceClient(model=model_id, token=token)

latest_date = future_idx[-1].strftime("%Y-%m")
latest_msi = float(forecast.values[-1])
latest_pmi = float(pmi.loc[future_idx[-1]])

prompt = (
    f"You are a financial analyst.\n"
    f"The latest MSI forecast for {latest_date} is {latest_msi:.1f}, "
    f"with PMI at {latest_pmi:.1f}.\n"
    "Write a brief analyst commentary on how MSI and PMI trends connect "
    "and what risks or opportunities they show for semiconductor manufacturing."
)

with st.spinner("Generating forecast commentary..."):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages,
            max_tokens=1000,
            temperature=0.7,
        )
        commentary = response.choices[0].message.content
    except Exception as e:
        commentary = f"❌ Error generating commentary: {e}"

st.subheader("📝 Automated MSI Forecast Analysis")
st.write(commentary)
