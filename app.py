import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Finnie", layout="wide")

# --- STYLE ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0a192f, #000000);
        color: #E0F2F1;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5 { color: #7dd3fc; font-weight: 600; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stTextInput>div>input, .stTextArea textarea {
        background-color: #111827;
        color: #e0f2f1;
        border: 1px solid #1e3a8a;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        transform: scale(1.02);
    }
    .result-box {
        background-color: #0f172a;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- INIT ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- SIDEBAR NAV ---
st.sidebar.title("Finnie")
page = st.sidebar.radio("Navigate", ["Portfolio Generator", "Ticker Analysis"])
st.sidebar.markdown("---")
st.sidebar.caption("Built for the Yale Entrepreneurial Society Fellowship")

# --- FUNCTIONS ---

def ai_generate_portfolio(goal_text):
    system_prompt = "You are Finnie, an AI investing assistant for beginner investors."
    user_prompt = f"""
    Based on this user's investing goals: "{goal_text}",
    build a sample portfolio with 5–8 assets (stocks and crypto).
    Include columns: Name, Ticker, Type (Stock/Crypto), and Allocation (%).
    Ensure total allocation = 100%.
    Finish with a short explanation of your reasoning.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def ai_analyze_ticker(symbol, summary_text):
    system_prompt = "You are Finnie, an educational AI assistant that analyzes financial assets in clear, friendly language."
    user_prompt = f"""
    Provide a short, easy-to-understand analysis for {symbol}.
    Use the market data below for context:
    {summary_text}
    Include:
    1. Overview
    2. Key performance trends
    3. Risks / volatility factors
    4. Long-term outlook
    Avoid financial advice.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
    )
    return response.choices[0].message.content


def ai_generate_insight(symbol, recent_return):
    """Extra concise insight about the asset."""
    prompt = f"""
    You are Finnie, an AI investing assistant.
    Give one short, high-level insight (1-2 sentences max) about {symbol}, 
    based only on this info: 6-month return of {recent_return}%.
    The tone should be analytical but friendly, with no financial advice.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content


def fetch_ticker_data(symbol):
    """More reliable data fetch using yfinance.history()"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")
        if data.empty:
            return None
        return data["Close"]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def summarize_ticker(symbol):
    """Pulls basic ticker info for snapshot"""
    info = {}
    try:
        yft = yf.Ticker(symbol)
        meta = yft.info
        info = {
            "Company": meta.get("shortName"),
            "Sector": meta.get("sector"),
            "Market Cap": meta.get("marketCap"),
            "PE Ratio": meta.get("trailingPE"),
            "52W High": meta.get("fiftyTwoWeekHigh"),
            "52W Low": meta.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        print(f"Summary fetch error: {e}")
    return info


# --- PAGE 1: PORTFOLIO GENERATOR ---
if page == "Portfolio Generator":
    st.title("Finnie: AI Portfolio Generator")
    st.markdown("Describe your investing goals, and Finnie will build a suggested stock and crypto portfolio.")

    goal = st.text_area(
        "Enter your goals:",
        height=120,
        placeholder="Example: I'm looking for moderate risk and long-term growth in tech and renewable energy.",
    )

    if st.button("Generate Portfolio"):
        if goal.strip() == "":
            st.error("Please describe your goals first.")
        else:
            with st.spinner("Generating your portfolio..."):
                ai_response = ai_generate_portfolio(goal)
                st.markdown("### Suggested Portfolio")
                st.markdown(f"<div class='result-box'>{ai_response}</div>", unsafe_allow_html=True)

# --- PAGE 2: TICKER ANALYSIS ---
elif page == "Ticker Analysis":
    st.title("Finnie: Ticker Analysis")
    st.markdown("Enter a stock or crypto ticker to get AI insights and recent price data.")
    symbol = st.text_input("Enter Ticker (e.g. AAPL, TSLA, BTC-USD):").upper()

    if st.button("Analyze"):
        if not symbol:
            st.error("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Fetching and analyzing {symbol}..."):
                data = fetch_ticker_data(symbol)
                if data is None or data.empty:
                    st.error(f"Could not fetch data for {symbol}. Try again or check your connection.")
                else:
                    info = summarize_ticker(symbol)
                    recent_return = round(((data[-1] / data[0]) - 1) * 100, 2)

                    st.markdown("### Market Snapshot")
                    if info:
                        st.dataframe(pd.DataFrame([info]).T.rename(columns={0: "Value"}))
                    else:
                        st.info("No company info found for this ticker.")

                    st.markdown(f"**6-Month Return:** {recent_return}%")
                    st.markdown("### Price Trend")
                    fig = px.line(data, x=data.index, y=data.values, labels={"x": "Date", "y": "Price (USD)"})
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # --- AI Analysis ---
                    summary_text = f"{symbol} 6-month return: {recent_return}% | Latest price: ${data[-1]:.2f}"
                    ai_analysis = ai_analyze_ticker(symbol, summary_text)
                    st.markdown("### AI Analysis")
                    st.markdown(f"<div class='result-box'>{ai_analysis}</div>", unsafe_allow_html=True)

                    # --- NEW: AI Insight ---
                    ai_insight = ai_generate_insight(symbol, recent_return)
                    st.markdown("### Finnie Insight")
                    st.markdown(f"<div class='result-box'>{ai_insight}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Finnie is for educational use only. Not financial advice.")
