# app.py
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from datetime import datetime, timedelta

# --- Page setup ---
st.set_page_config(page_title="Finnie - AI Portfolio Builder", layout="wide", page_icon="💎")

# Initialize OpenAI client with secret from Streamlit
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Styling (dark / cool) ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f16;
        color: #e0f2f1;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 { color: #7dd3fc !important; }
    .stButton>button { background-color: #2563eb; color: white; border-radius: 10px; }
    .stButton>button:hover { background-color: #1d4ed8; }
    .stTextInput>div>input { background: #071025; color: #e0f2f1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.title("Finnie")
st.caption("AI assistant that builds portfolios and provides ticker-level analysis.")
st.divider()

# --- Section 1: Goals -> Portfolio (existing feature) ---
st.subheader("Generate a portfolio from your goals")
goal = st.text_area(
    "Describe your investing goals. Example: 'I want a moderate-risk portfolio focused on tech and clean energy, investing horizon 5 years.'",
    height=100,
)

def get_portfolio_from_ai(goal_text: str) -> str:
    system_msg = (
        "You are Finnie, an AI investing assistant for student users. "
        "Provide clear, educational, non-prescriptive portfolio suggestions."
    )
    prompt = (
        f"Based on this goal: \"{goal_text}\", suggest a diversified portfolio with both stocks and crypto. "
        "Provide 5-8 assets with percentage allocations that sum to 100%. "
        "For each asset show: Ticker, Name, Type (Stock/Crypto), Allocation (%). "
        "Then add a short 2-3 sentence rationale. Use a simple plain-text table or bullet list."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content

def safe_extract_tickers(text: str):
    # crude but practical extraction: uppercase words up to 5 chars or common crypto symbols
    tickers = set()
    for token in text.replace("|", " ").replace(",", " ").split():
        tok = token.strip()
        if tok.isupper() and 1 <= len(tok) <= 5:
            tickers.add(tok)
    # common crypto symbols are usually uppercase but sometimes not; still captured above
    return list(tickers)

if st.button("Generate Portfolio") and goal.strip():
    with st.spinner("Building portfolio..."):
        ai_output = get_portfolio_from_ai(goal)
        st.markdown("### AI Portfolio Suggestion")
        st.text(ai_output)

        # show parsed table if possible
        tickers = safe_extract_tickers(ai_output)
        if tickers:
            try:
                prices = yf.download(tickers, period="5d")["Adj Close"].tail(1).T
                price_df = prices.reset_index()
                price_df.columns = ["Ticker", "Price (USD)"]
                st.dataframe(price_df, use_container_width=True)
            except Exception:
                st.info("Could not fetch price snapshot for all tickers.")

st.markdown("---")

# --- Section 2: Ticker analysis (new feature) ---
st.subheader("Ticker analysis")
ticker_input = st.text_input("Enter a ticker (stock symbol like AAPL or crypto like BTC):", value="")

def is_probable_crypto(ticker: str) -> bool:
    # heuristic: common cryptos or if yfinance fails to return basic info
    crypto_candidates = {"BTC", "ETH", "SOL", "ADA", "BNB", "DOGE", "DOT", "AVAX", "XRP", "LTC"}
    return ticker.upper() in crypto_candidates

def fetch_stock_data(ticker: str, days: int = 90):
    try:
        tk = ticker.upper()
        data = yf.download(tk, period=f"{days}d", interval="1d")["Adj Close"].dropna()
        info = yf.Ticker(tk).info
        return data, info
    except Exception:
        return None, None

def fetch_crypto_data_by_coingecko_id(cg_id: str, days: int = 90):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        result = r.json()
        # result["prices"] = [ [timestamp, price], ... ]
        prices = pd.DataFrame(result["prices"], columns=["ts", "price"])
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms")
        prices = prices.set_index("date")["price"].resample("1D").mean().ffill()
        return prices
    except Exception:
        return None

def map_symbol_to_coingecko_id(symbol: str):
    # a tiny mapping for common crypto tickers; extend as needed
    mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "BNB": "binancecoin",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "XRP": "ripple",
        "LTC": "litecoin",
    }
    return mapping.get(symbol.upper())

def compute_metrics(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return {}
    latest = float(series.iloc[-1])
    returns_30 = float((series.iloc[-1] / series.iloc[-31] - 1) * 100) if len(series) > 31 else None
    returns_90 = float((series.iloc[-1] / series.iloc[0] - 1) * 100) if len(series) >= 2 else None
    daily_ret = series.pct_change().dropna()
    volatility_30 = float(daily_ret.tail(30).std() * np.sqrt(252) * 100) if len(daily_ret) >= 30 else None
    return {
        "latest_price": latest,
        "return_30d_pct": returns_30,
        "return_total_pct": returns_90,
        "annualized_vol_pct": volatility_30,
    }

def get_ticker_analysis_with_ai(ticker_symbol: str, market_context: dict, extra_context: str = "") -> str:
    system_msg = "You are Finnie, an educational AI investing assistant. Do not provide legal/financial advice. Provide clear, educational analysis with risk indicators."
    prompt = (
        f"Provide a detailed analysis for the ticker '{ticker_symbol}'.\n\n"
        "Market data (use this to ground your analysis):\n"
    )
    for k, v in market_context.items():
        prompt += f"- {k}: {v}\n"
    if extra_context:
        prompt += f"\nAdditional context:\n{extra_context}\n"
    prompt += (
        "\nPlease produce:\n"
        "1) Brief summary (1-2 sentences)\n"
        "2) Key drivers and catalysts\n"
        "3) Risks / what to watch\n"
        "4) Suggested time horizons that fit this asset type\n"
        "5) A short non-prescriptive educational takeaway\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
    )
    return resp.choices[0].message.content

if st.button("Analyze Ticker") and ticker_input.strip():
    sym = ticker_input.strip().upper()
    with st.spinner(f"Fetching market data for {sym}..."):
        # Try as stock first
        stock_series, stock_info = fetch_stock_data(sym, days=180)
        if stock_series is not None and len(stock_series) >= 2:
            metrics = compute_metrics(stock_series)
            market_context = {
                "Type": "Stock",
                "Latest close (USD)": metrics.get("latest_price"),
                "30-day return (%)": metrics.get("return_30d_pct"),
                "90-day return (%)": metrics.get("return_total_pct"),
                "Annualized volatility (%)": metrics.get("annualized_vol_pct"),
                "Company shortName": stock_info.get("shortName") if isinstance(stock_info, dict) else None,
                "Sector": stock_info.get("sector") if isinstance(stock_info, dict) else None,
                "Market cap": stock_info.get("marketCap") if isinstance(stock_info, dict) else None,
            }
            st.markdown(f"### {sym} — Stock overview")
            st.dataframe(pd.DataFrame([market_context]).T.rename(columns={0: "Value"}), use_container_width=True)

            # Price chart
            fig = px.line(stock_series.reset_index(), x="Date", y=sym if sym in stock_series.columns else stock_series.name,
                          labels={"value": "Price (USD)"})
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

            # AI analysis
            ai_analysis = get_ticker_analysis_with_ai(sym, market_context)
            st.markdown("#### AI Analysis")
            st.text(ai_analysis)

        else:
            # Try as crypto via CoinGecko mapping
            cg_id = map_symbol_to_coingecko_id(sym)
            if cg_id:
                crypto_series = fetch_crypto_data_by_coingecko_id(cg_id, days=180)
                if crypto_series is not None and len(crypto_series) >= 2:
                    metrics = compute_metrics(crypto_series)
                    market_context = {
                        "Type": "Crypto",
                        "CoinGecko id": cg_id,
                        "Latest price (USD)": metrics.get("latest_price"),
                        "30-day return (%)": metrics.get("return_30d_pct"),
                        "180-day return (%)": metrics.get("return_total_pct"),
                        "Annualized volatility (%)": metrics.get("annualized_vol_pct"),
                    }
                    st.markdown(f"### {sym} — Crypto overview")
                    st.dataframe(pd.DataFrame([market_context]).T.rename(columns={0: "Value"}), use_container_width=True)

                    # Price chart
                    fig = px.line(crypto_series.reset_index(), x="date", y="price", labels={"price": "Price (USD)", "date": "Date"})
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    ai_analysis = get_ticker_analysis_with_ai(sym, market_context)
                    st.markdown("#### AI Analysis")
                    st.text(ai_analysis)
                else:
                    st.error("Could not fetch crypto price series from CoinGecko.")
            else:
                # final attempt: try yfinance with appended suffixes (common for other exchanges)
                st.error("Ticker not found as a US stock or recognized crypto symbol. Try a different ticker or use a common crypto (BTC, ETH).")

st.markdown("---")
st.caption("Finnie provides educational analysis only. This is not financial advice.")

# --- requirements note ---
# Make sure your requirements.txt contains:
# streamlit
# openai
# pandas
# numpy
# yfinance
# plotly
# requests
