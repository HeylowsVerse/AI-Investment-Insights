import argparse
import json
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path
from typing import Union

import feedparser
import requests
import yfinance as yf
from bs4 import BeautifulSoup

SEMICONDUCTOR_COMPANIES = {
    "ADI": "Analog Devices", "AIP": "Arteris", "ALAB": "Astera Labs",
    "ALGM": "Allegro Microsystems", "ALMU": "Altium", "AMD": "Advanced Micro Devices",
    "AOSL": "Alpha and Omega Semiconductor", "ARM": "Arm Holdings",
    "ASX": "ASE Technology", "AVGO": "Broadcom", "CEVA": "CEVA Inc",
    "CRD": "Cerdelga", "CRUS": "Cirrus Logic", "DIOD": "Diodes Inc",
    "GCTS": "GCT Semiconductor", "GFS": "GlobalFoundries", "GSI": "GSI Technology",
    "HIMX": "Himax Technologies", "ICG": "Intchains Group", "IMOS": "ChipMOS Technologies",
    "INTC": "Intel", "LASR": "nLIGHT", "LEDS": "SemiLEDs", "LSCC": "Lattice Semiconductor",
    "MCHP": "Microchip Technology", "MOBX": "Mobix Labs", "MPWR": "Monolithic Power Systems",
    "MRAM": "Everspin Technologies", "MRVL": "Marvell Technology",
    "MU": "Micron Technology", "MXL": "MaxLinear", "MX": "MagnaChip Semiconductor",
    "NAN": "Nanophase Technologies", "NVDA": "NVIDIA", "NVEC": "NVE Corporation",
    "NVTS": "Navitas Semiconductor", "NXPI": "NXP Semiconductors", "ON": "ON Semiconductor",
    "PII": "Polaris", "POWI": "Power Integrations", "PRSO": "Peraso", "PXLW": "Pixelworks",
    "QCOM": "Qualcomm", "QRVO": "Qorvo", "QUIK": "QuickLogic", "RMBS": "Rambus",
    "SIMO": "Silicon Motion", "SITM": "SiTime", "SKY": "SkyWater Technology",
    "SLAB": "Silicon Labs", "SMTC": "Semtech", "SQNS": "Sequans Communications",
    "STM": "STMicroelectronics", "SWKS": "Skyworks Solutions", "SYNA": "Synaptics",
    "TSEM": "Tower Semiconductor", "TSM": "Taiwan Semiconductor", "TXN": "Texas Instruments",
    "UMC": "United Microelectronics", "VLN": "Valens Semiconductor",
    "VSH": "Vishay Intertechnology", "WKEY": "WISeKey", "WOLF": "Wolfspeed"
}

CIK_MAP = {
    "NVDA": "0001045810", "AVGO": "0001730168", "TSM": "0001046179",
    "AMD": "0000002488", "TXN": "0000097476", "QCOM": "0000804328",
    "MU": "0000723125", "INTC": "0000050863", "ADI": "0000006281",
    "MRVL": "0001330474", "NXPI": "0001413447", "MCHP": "0000827054",
    "MPWR": "0001109355", "STM": "0001651027", "ON": "0001097864",
    "GFS": "0001091277", "ASX": "0001117043", "UMC": "0000789582",
    "SWKS": "0001120409", "QRVO": "0001522994", "LSCC": "0001089052",
    "RMBS": "0001349804", "ALGM": "0000950136", "CRUS": "0000910415",
    "TSEM": "0000928876", "SLAB": "0001466903", "SMTC": "0000750492",
    "PII": "0001005693", "POWI": "0001682092", "DIOD": "0000826705",
    "SYNA": "0001113619", "SIMO": "0000895420", "VSH": "0000796275",
    "HIMX": "0001411723", "MXL": "0000953499", "NVTS": "0001750699",
    "LASR": "0001576374", "AOSL": "0000866012", "IMOS": "0000839809",
    "CEVA": "0001396501", "SKY": "0001606108", "AIP": "0001595557",
    "NVEC": "0001528184", "VLN": "0001623430", "ALMU": "0001569872",
    "WOLF": "0001647299", "MRAM": "0001628258", "MX": "0001587860",
    "NAN": "0001743500", "ICG": "0001394629", "GSI": "0001156469",
    "QUIK": "0001149544", "GCTS": "0001737762", "MOBX": "0001782102",
    "WKEY": "0001250374", "PXLW": "0001351794", "SQNS": "0001219669",
    "LEDS": "0001735478", "PRSO": "0001689326", "CRD": "0001716731",
    "ALAB": "0001781815"
}


DEFAULT_DOWNLOAD_DIR = Path.home() / "Downloads"


def save_json(data, filename, output_dir=DEFAULT_DOWNLOAD_DIR):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {file_path}")


def fetch_stock_data(ticker: str, days: int):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if df.empty:
        raise ValueError("No stock data returned")
    df = df.reset_index()
    df["Date"] = df["Date"].astype(str)
    return df.to_dict(orient="records")


def fetch_filing_url(ticker: str):
    cik = CIK_MAP.get(ticker.upper())
    if not cik:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None
    data = r.json()
    for i, form in enumerate(data["filings"]["recent"]["form"]):
        if form in {"ARS", "10-K", "10-Q", "8-K"}:
            acc = data["filings"]["recent"]["accessionNumber"][i].replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{acc}-index.htm"
    return None


def download_filing_text(url: str) -> str:
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text()


def fetch_google_news(query: str):
    safe_query = quote(query + " semiconductor when:7d")
    feed_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    return [
        {"title": e.title, "link": e.link, "published": e.published, "summary": e.summary}
        for e in feed.entries
    ]


def main(
    ticker: str,
    days: int,
    output_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
) -> dict:
    """Scrape data and save JSON files.

    Returns a dictionary with the scraped data so callers (e.g. the
    Streamlit app) can offer download buttons without reading the files
    back from disk.
    """
    output_dir = Path(output_dir).expanduser()
    result: dict = {}
    name = SEMICONDUCTOR_COMPANIES.get(ticker.upper(), ticker)

    stock = fetch_stock_data(ticker, days)
    save_json(stock, f"{ticker}_stock.json", output_dir)
    result["stock"] = stock

    url = fetch_filing_url(ticker)
    if url:
        text = download_filing_text(url)
        filing_data = {"ticker": ticker, "url": url, "report_text": text}
        save_json(filing_data, f"{ticker}_filing_latest.json", output_dir)
        result["filings"] = filing_data
    else:
        print("No recent SEC filing found.")
        result["filings"] = None

    news = fetch_google_news(name)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    save_json(news, f"{ticker}_news_{ts}.json", output_dir)
    result["news"] = news
    result["timestamp"] = ts

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape stock data, SEC filings and news for a company")
    parser.add_argument("ticker", help="Ticker symbol")
    parser.add_argument("--days", type=int, default=90, help="Number of past days for stock prices")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory to save JSON files (defaults to your Downloads folder)",
    )
    args = parser.parse_args()
    main(args.ticker.upper(), args.days, args.output_dir)
