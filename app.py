import streamlit as st
from openai import OpenAI
import pandas as pd
import yfinance as yf
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="Finnie - AI Portfolio Builder", layout="wide", page_icon="💹")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- APP HEADER ---
st.title("💹 Finnie")
st.caption("Your AI-powered investing buddy — generate a smart portfolio from your goals.")
st.divider()

# --- USER INPUT ---
st.subheader("🎯 Tell Finnie your investment goals")
goal = st.text_area(
    "Example: 'I want a medium-risk portfolio focused on tech stocks and some Bitcoin exposure.'",
    height=100
)

if st.button("Generate Portfolio 🚀") and goal.strip():
    with st.spinner("Finnie is building your personalized portfolio..."):
        # --- ASK OPENAI ---
        prompt = f"""
        You are Finnie, an AI investing assistant.
        Based on this user's goals, suggest a diversified portfolio with tickers (stocks + crypto).
        Include 5-10 assets with percentage allocations summing to 100%.
        Format your response as a table with columns: Asset, Ticker, Type (Stock/Crypto), Allocation (%).
        User goal: {goal}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        st.markdown("### 🧾 Finnie's Portfolio Suggestion")
        st.markdown(output)

        # --- OPTIONAL: Extract tickers and show basic price data ---
        tickers = []
        for line in output.splitlines():
            if "," in line or "|" in line:
                for word in line.split():
                    if word.isupper() and len(word) <= 5:
                        tickers.append(word.replace("|", "").replace(",", ""))

        tickers = list(set(tickers))
        if tickers:
            st.markdown("### 📊 Recent Price Trends")
            try:
                data = yf.download(tickers, period="3mo")["Adj Close"]
                data = data.fillna(method="ffill")
                fig = px.line(data, x=data.index, y=data.columns)
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Couldn’t fetch price data — some tickers may not be valid.")

else:
    st.info("💬 Enter your investing goals above, then click **Generate Portfolio**.")

# --- FOOTER ---
st.divider()
st.caption("Built with ❤️ by a high schooler for the Yale Entrepreneurial Society Fellowship.")
