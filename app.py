import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as yf_st
import yfinance as yf
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import anthropic

# -------------------------------------------------------------
# API KEYS CONFIGURATION (Secure Streamlit Secrets Lookup)
# -------------------------------------------------------------
MY_GEMINI_API_KEY = yf_st.secrets["GEMINI_API_KEY"]
MY_ANTHROPIC_API_KEY = yf_st.secrets["ANTHROPIC_API_KEY"]

# Page Layout Configuration
yf_st.set_page_config(
    page_title="Indian Index & Stock Options Quant Desk",
    page_icon="📊",
    layout="wide"
)

yf_st.title("📊 Indian Index & Stock Options Quant Desk (Multi-AI Failover)")
yf_st.markdown("Powered by **Python**, **yfinance**, **Google Gemini**, and **Anthropic Claude** fallback.")

# Sidebar Controls
yf_st.sidebar.header("Options Screener Controls")
analysis_mode = yf_st.sidebar.selectbox(
    "Select Target Market Segments", 
    ["Indices Only (Nifty, Sensex, Midcap)", "Top NSE Stock Options Only", "Comprehensive Desk Scan (Indices + Stocks)"]
)
run_button = yf_st.sidebar.button("Generate Option Setups 🚀", type="primary")

# Pydantic Schema
class OptionTradeRecommendation(BaseModel):
    instrument_name: str
    asset_type: str
    recommended_strategy: str
    market_outlook: str
    entry_trigger: float
    strikes_to_deploy: str
    max_reward_per_lot: str
    max_risk_per_lot: str
    probability_of_profit: str
    risk_assessment_details: str

class OptionsMasterReport(BaseModel):
    macro_market_summary: str
    trade_recommendations: list[OptionTradeRecommendation]


# CACHED MARKET DATA FETCHING
@yf_st.cache_data(ttl=300)
def fetch_market_data() -> dict:
    tickers = {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Nifty Midcap 50": "^NSEMDCP50",
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "Tata Motors": "TATAMOTORS.NS"
    }
    data_store = {}
    
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        trading_days = 252
        hv_20 = float(df["Log_Ret"].tail(20).std() * np.sqrt(trading_days) * 100)
        
        latest_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        pct_change = ((latest_close - prev_close) / prev_close) * 100

        data_store[name] = {
            "latest_close": round(latest_close, 2),
            "change_pct": round(pct_change, 2),
            "hv_20d": round(hv_20, 2),
            "20d_sma": round(float(df["Close"].tail(20).mean()), 2)
        }
    return data_store


if run_button:
    if not MY_GEMINI_API_KEY or not MY_ANTHROPIC_API_KEY:
        yf_st.error("Please ensure both `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` are saved in Streamlit Secrets!")
    else:
        with yf_st.spinner("Pulling market matrices and initializing Multi-AI routing..."):
            try:
                market_data = fetch_market_data()
                
                prompt = f"""
You are an expert quantitative derivatives portfolio manager specializing in Indian index options (Nifty, Sensex, Midcap) and high-liquidity NSE stock options.
Analyze the following underlying price metrics and 20-day historical volatilities:
{market_data}

Current Selection Mode Requested by User: {analysis_mode}

Provide a rigorous options recommendation report matching this mode. Include exact strike selections, defined credit/risk structures, maximum reward, maximum risk, and an estimated probability of profit (accuracy percentage).
You must output strictly valid JSON matching this schema:
{OptionsMasterReport.model_json_schema()}
"""

                response_text = None
                used_model = ""

                # --- TRY GEMINI FIRST ---
                try:
                    client = genai.Client(api_key=MY_GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json",
                            response_schema=OptionsMasterReport,
                        ),
                    )
                    response_text = response.text
                    used_model = "Google Gemini 3.8 Flash"
                except Exception as gemini_err:
                    yf_st.warning(f"Gemini encountered high traffic or error ({gemini_err}). Failing over to Anthropic Claude...")
                    
                    # --- FALLBACK TO ANTHROPIC CLAUDE ---
                    anth_client = anthropic.Anthropic(api_key=MY_ANTHROPIC_API_KEY)
                    claude_message = anth_client.messages.create(
                        model="claude-3-5-sonnet",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt + "\n\nRespond with ONLY valid JSON block, no extra text."}]
                    )
                    raw_content = claude_message.content[0].text
                    
                    # Clean markdown code blocks if returned by Claude
                    if "```json" in raw_content:
                        raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw_content:
                        raw_content = raw_content.split("```")[1].split("```")[0].strip()
                        
                    response_text = raw_content
                    used_model = "Anthropic Claude 3.5 Sonnet (Failover Active)"

                # Validate and parse response into Pydantic model
                report = OptionsMasterReport.model_validate_json(response_text)

                yf_st.success(f"Options intelligence report successfully compiled via **{used_model}**!")

                yf_st.subheader("🌐 Macro Market Context & Volatility Environment")
                yf_st.info(report.macro_market_summary)

                yf_st.subheader("🎯 High-Probability Option Recommendations & Risk Profile")
                
                for rec in report.trade_recommendations:
                    with yf_st.expander(f"📌 [{rec.asset_type.upper()}] {rec.instrument_name} — Strategy: {rec.recommended_strategy}", expanded=True):
                        col1, col2, col3 = yf_st.columns(3)
                        with col1:
                            yf_st.metric("Estimated Accuracy (Win Rate)", rec.probability_of_profit)
                            yf_st.markdown(f"**Trigger Price / Spot:** {rec.entry_trigger}")
                        with col2:
                            yf_st.markdown(f"**Max Reward per Lot:** {rec.max_reward_per_lot}")
                            yf_st.markdown(f"**Max Risk per Lot:** {rec.max_risk_per_lot}")
                        with col3:
                            yf_st.markdown(f"**Suggested Strikes:** {rec.strikes_to_deploy}")
                        
                        yf_st.markdown("---")
                        yf_st.markdown(f"**Market Outlook:** {rec.market_outlook}")
                        yf_st.markdown(f"**Risk Assessment & Management:** {rec.risk_assessment_details}")

                table_rows = [{
                    "Instrument": r.instrument_name,
                    "Type": r.asset_type,
                    "Strategy": r.recommended_strategy,
                    "Strikes": r.strikes_to_deploy,
                    "Accuracy / Win Rate": r.probability_of_profit,
                    "Max Risk": r.max_risk_per_lot
                } for r in report.trade_recommendations]
                
                df_summary = pd.DataFrame(table_rows)
                yf_st.subheader("📋 Quick Reference Table")
                yf_st.dataframe(df_summary, use_container_width=True)

                csv_bytes = df_summary.to_csv(index=False).encode('utf-8')
                yf_st.download_button(
                    label="📥 Download Options Strategy CSV Log",
                    data=csv_bytes,
                    file_name="options_recommendations.csv",
                    mime="text/csv"
                )

            except Exception as e:
                yf_st.error(f"Both AI providers experienced connection issues. Details: {e}")
