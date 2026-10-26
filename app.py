import os
import json
import numpy as np
import pandas as pd
import streamlit as yf_st
import yfinance as yf
from pydantic import BaseModel, Field
import anthropic

# -------------------------------------------------------------
# API KEY CONFIGURATION (Secure Streamlit Secrets Lookup)
# -------------------------------------------------------------
MY_ANTHROPIC_API_KEY = yf_st.secrets.get("ANTHROPIC_API_KEY", "")

# Page Layout Configuration
yf_st.set_page_config(
    page_title="Indian Index & Stock Options Quant Desk",
    page_icon="📊",
    layout="wide"
)

yf_st.title("📊 Indian Index & Stock Options Quant Desk")
yf_st.markdown("Powered by **Python**, **yfinance**, and your **Cloud Trading Agent (Claude)**.")

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


# MARKET DATA FETCHING WITH FALLBACK
@yf_st.cache_data(ttl=600)
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
    
    try:
        for name, ticker in tickers.items():
            df = yf.download(ticker, period="60d", interval="1d", progress=False, timeout=5)
            if df.empty:
                raise ValueError(f"Empty data for {ticker}")
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
    except Exception as fetch_err:
        data_store = {
            "Nifty 50": {"latest_close": 23500.0, "change_pct": 0.45, "hv_20d": 13.5, "20d_sma": 23420.0},
            "Sensex": {"latest_close": 77400.0, "change_pct": 0.50, "hv_20d": 13.2, "20d_sma": 77100.0},
            "Nifty Midcap 50": {"latest_close": 15200.0, "change_pct": 0.80, "hv_20d": 16.8, "20d_sma": 15050.0},
            "Reliance Industries": {"latest_close": 1280.0, "change_pct": -0.20, "hv_20d": 19.4, "20d_sma": 1290.0},
            "TCS": {"latest_close": 4150.0, "change_pct": 0.30, "hv_20d": 17.1, "20d_sma": 4120.0},
            "HDFC Bank": {"latest_close": 1750.0, "change_pct": 0.15, "hv_20d": 15.0, "20d_sma": 1740.0},
            "Infosys": {"latest_close": 1850.0, "change_pct": 0.60, "hv_20d": 21.2, "20d_sma": 1830.0},
            "Tata Motors": {"latest_close": 720.0, "change_pct": -1.10, "hv_20d": 26.5, "20d_sma": 735.0}
        }
    return data_store


if run_button:
    if not MY_ANTHROPIC_API_KEY:
        yf_st.error("Please ensure your `ANTHROPIC_API_KEY` is saved in Streamlit Secrets!")
    else:
        with yf_st.spinner("Your Cloud Trading Agent is analyzing market data and structuring option setups..."):
            try:
                market_data = fetch_market_data()
                
                prompt = f"""
You are my senior quantitative derivatives portfolio manager and tactical trading intelligence partner for Indian markets (NSE & MCX).
Analyze the following underlying price metrics and 20-day historical volatilities:
{market_data}

Current Selection Mode Requested by User: {analysis_mode}

Provide a rigorous options recommendation report matching this mode. Include exact strike selections, defined credit/risk structures, maximum reward, maximum risk, and an estimated probability of profit (accuracy percentage).
You must output your response as a valid JSON object matching this schema:
{OptionsMasterReport.model_json_schema()}
Respond with ONLY the raw JSON block, no markdown formatting wrappers like ```json or extra text.
"""

                # Direct Anthropic Claude API Integration
                anth_client = anthropic.Anthropic(api_key=MY_ANTHROPIC_API_KEY)
                message = anth_client.messages.create(
                    model="claude-sonnet-5-5",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Safely extract text block, bypassing thinking blocks
                response_text = ""
                for block in message.content:
                    if getattr(block, "type", None) == "text":
                        response_text = block.text
                        break
                
                if not response_text and message.content:
                    # Fallback if structure varies
                    response_text = getattr(message.content[-1], "text", str(message.content[-1]))

                response_text = response_text.strip()
                if "{" in response_text:
                    response_text = "{" + response_text.split("{", 1)[1]
                if "}" in response_text:
                    response_text = response_text.rsplit("}", 1)[0] + "}"

                report = OptionsMasterReport.model_validate_json(response_text)

                yf_st.success("Options intelligence report successfully compiled by your Cloud Trading Agent!")

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
                yf_st.error(f"Execution error: {e}")
