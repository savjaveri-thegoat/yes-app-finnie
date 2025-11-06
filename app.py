import streamlit as st
import openai
import yfinance as yf
import requests
import pandas as pd
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Finnie | AI Investing Assistant", page_icon="💎", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
/* Dark theme with cool accent colors */
[data-testid="stAppViewContainer"] {
    background-color: #0b0f16;
    color: #e0f2f1;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3, h4 {
    color: #7dd3fc !important;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.03);
}
a, a:visited {
    color: #93c5fd;
}
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("💎 Finnie")
st.subheader("Your AI Investing Assistant")
st.caption("_Dream it. Plan it. Build your future with data-driven investing._")

# --- USER INPUT ---
goal = st.text_area(
    "🧭 Describe your investing goals:",
    placeholder="e.g. I'm 17, saving for college in 5 years, moderate risk, interested in tech and crypto."
)

# --- API KEY ---
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# --- MAIN FUNCTION ---
def get_portfolio_from_ai(goal_text):
    prompt = f"""
    You are Finnie, an AI investing assistant for a high school student.
    Based on this goal: '{goal_text}', create a diversified investment portfolio with:
    - Stocks and crypto
    - % allocation for each asset (total 100%)
    - 2–3 sentence rationale.
    Format:
    Portfolio:
    - [TICKER] - [Name] - [Type: Stock/Crypto] - [Percentage]%
    Rationale: [text]
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- PARSE PORTFOLIO TEXT ---
def parse_portfolio(text):
    lines = text.splitlines()
    assets = []
    for line in lines:
        if "-" in line and "%" in line:
            try:
                parts = [p.strip() for p in line.split("-")]
                ticker = parts[0].replace("•", "").strip()
                name = parts[1].strip()
                type_ = parts[2].strip()
                pct = float(parts[3].replace("%", "").strip())
                assets.append({"Ticker": ticker, "Name": name, "Type": type_, "Allocation %": pct})
            except:
                continue
    return pd.DataFrame(assets)

# --- FETCH PRICES ---
def get_prices(df):
    prices = []
    for _, row in df.iterrows():
        if row["Type"].lower() == "stock":
            try:
                data = yf.Ticker(row["Ticker"]).history(period="1d")
                price = round(data["Close"].iloc[-1], 2)
            except:
                price = None
        else:  # crypto
            try:
                cg_id = row["Name"].split()[0].lower()
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
                res = requests.get(url).json()
                price = list(res.values())[0]["usd"]
            except:
                price = None
        prices.append(price)
    df["Price (USD)"] = prices
    return df

# --- MAIN LOGIC ---
if st.button("✨ Generate My Portfolio"):
    if not goal.strip():
        st.warning("Please describe your goals first!")
    else:
        with st.spinner("🔍 Finnie is analyzing your goals..."):
            ai_text = get_portfolio_from_ai(goal)
            st.subheader("📊 Your AI Portfolio")
            st.write(ai_text)

            df = parse_portfolio(ai_text)
            if not df.empty:
                df = get_prices(df)
                st.dataframe(df, use_container_width=True)

                fig = px.pie(
                    df,
                    values="Allocation %",
                    names="Ticker",
                    title="💠 Portfolio Allocation",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not parse portfolio — check AI output format.")

st.markdown("---")
st.caption("⚠️ Finnie is for educational purposes only — not financial advice.")
