"""
Indian Index & Stock Options Quant Desk (Read-Only Analysis Tool)
"""

from __future__ import annotations

import hmac
import json
import math
from datetime import datetime, time as dtime
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL = "claude-sonnet-5-5"
IST = ZoneInfo("Asia/Kolkata")
MAX_DATA_AGE_DAYS = 4              # Covers weekends + one holiday
PARTIAL_BAR_CUTOFF = dtime(15, 45) # Before this, today's daily bar is incomplete

# Exchange-aligned lot sizes (MUST be verified periodically against nseindia.com / bseindia.com)
LOT_SIZE_SOURCE = "Exchange guidelines / Broker terminal (UNVERIFIED by app - check live)"
INSTRUMENTS: dict[str, dict] = {
    "Nifty 50":              {"symbol": "^NSEI",       "lot": 25,   "kind": "Index"},
    "Bank Nifty":            {"symbol": "^NSEBANK",    "lot": 30,   "kind": "Index"},
    "Sensex":                {"symbol": "^BSESN",      "lot": 10,   "kind": "Index"},
    "Reliance Industries":   {"symbol": "RELIANCE.NS", "lot": 250,  "kind": "Stock"},
    "TCS":                   {"symbol": "TCS.NS",      "lot": 175,  "kind": "Stock"},
    "HDFC Bank":             {"symbol": "HDFCBANK.NS", "lot": 550,  "kind": "Stock"},
    "Infosys":               {"symbol": "INFY.NS",     "lot": 400,  "kind": "Stock"},
    "Tata Motors":           {"symbol": "TATAMOTORS.NS", "lot": 1400, "kind": "Stock"},
}

MODES = {
    "Indices Only (Nifty, Bank Nifty, Sensex)": ("Index",),
    "Top NSE Stock Options Only": ("Stock",),
    "Comprehensive Desk Scan (Indices + Stocks)": ("Index", "Stock"),
}

# ----------------------------------------------------------------------------
# Strict Pydantic Schema (Forced Tool Use)
# ----------------------------------------------------------------------------
class Recommendation(BaseModel):
    instrument_name: str = Field(description="EXACT name as given in market_data keys")
    asset_type: Literal["Index", "Stock"]
    recommended_strategy: str = Field(description="e.g. 'Bear put spread' or 'None'")
    market_outlook: str = Field(description="2-3 sentences, max")
    entry_condition: str = Field(description="What must happen first, e.g. a closed 15m/daily candle level")
    invalidation_level: Optional[float] = Field(
        default=None, description="Index/stock level that cancels the idea; null if no idea")
    strikes_to_deploy: str = Field(
        default="Not specified",
        description="Illustrative only. Must be confirmed against the live option chain")
    conditional_idea: str = Field(description="What would make a trade valid; 'None' if flat")
    risk_assessment_details: str = Field(description="2-3 sentences, include main risk")
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    verdict: Literal["TAKE_TRADE", "STAY_FLAT"]


class OptionsMasterReport(BaseModel):
    macro_market_summary: str = Field(
        description="Short summary. Mark each fact VERIFIED/UNVERIFIED/HYPOTHESIS/UNKNOWN")
    unknowns: list[str] = Field(description="Important items that were not provided")
    trade_recommendations: list[Recommendation]


SYSTEM_PROMPT = """You are a senior quantitative derivatives portfolio manager and tactical risk partner for Indian markets (NSE/MCX).
You are READ-ONLY: you never place trades and you never ask for broker logins or PINs. The user makes every decision.

HARD RULES
1. Use ONLY the data in the user message (market_data, live_web_research, user inputs). Never invent prices, IV, OI, premiums, flows, news or probabilities.
   Anything missing is UNKNOWN. Do NOT output win rates, probabilities of profit, premiums or max profit/loss.
2. Label facts: VERIFIED (computed from market_data, or two independent sources agree, with a time), UNVERIFIED (single web source, no timestamp, or user-supplied),
   HYPOTHESIS (your inference), UNKNOWN. Items in live_web_research are UNVERIFIED unless two sources agree. User-typed macro values override web values.
3. HV-20 is realised volatility, not implied volatility. Do not say options are cheap or expensive without implied volatility data.
4. If data_status is not OK for an instrument, or ideas_gate is CLOSED, or the option chain is NOT PROVIDED, the verdict must be STAY_FLAT.
5. If evidence is mixed or noisy, the verdict is STAY_FLAT. A flat verdict is a good result.
6. Every TAKE_TRADE needs an entry_condition and an invalidation_level. Prefer defined-risk structures. Never describe any idea as 'high probability' or 'safe'.
7. Use the exact instrument names from market_data. Keep text short, in simple English.
8. Call the tool submit_report exactly once."""


# ----------------------------------------------------------------------------
# Time / Session Helpers (IST)
# ----------------------------------------------------------------------------
def now_ist() -> datetime:
    return datetime.now(IST)


def session_label(now: datetime) -> str:
    if now.weekday() >= 5:
        return "CLOSED_WEEKEND"
    t = now.time()
    if t < dtime(9, 15):
        return "BEFORE_OPEN"
    if t < dtime(9, 30):
        return "FIRST_15M_CANDLE_OPEN"
    if t <= dtime(15, 30):
        return "OPEN"
    return "CLOSED_AFTER_MARKET"


def ideas_gate(now: datetime) -> str:
    label = session_label(now)
    if label in ("BEFORE_OPEN", "FIRST_15M_CANDLE_OPEN"):
        return "CLOSED"
    return "OPEN"


# ----------------------------------------------------------------------------
# Robust Market Data Fetching (Per-Ticker Isolation)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _download(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="90d", interval="1d", progress=False,
                     auto_adjust=True, timeout=10)
    if df is None or df.empty:
        raise ValueError("empty download")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError("no Close column")
    return df[["Close"]].dropna()


def compute_metrics(df: pd.DataFrame, now: datetime) -> dict:
    close = df["Close"].dropna()
    dropped_partial = False
    
    # Drop today's live bar if market is still running / bar is incomplete
    if len(close) and close.index[-1].date() == now.date() and now.time() < PARTIAL_BAR_CUTOFF:
        close = close.iloc[:-1]
        dropped_partial = True
        
    if len(close) < 21:
        raise ValueError(f"only {len(close)} closes, need 21")

    log_ret = np.log(close / close.shift(1)).dropna().tail(20) # 21 closes -> 20 returns
    if len(log_ret) < 20:
        raise ValueError("fewer than 20 returns")

    hv20 = float(log_ret.std(ddof=1) * math.sqrt(252) * 100)
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_bar_date = close.index[-1].date()
    age_days = (now.date() - last_bar_date).days
    
    return {
        "last_completed_close": round(last_close, 2),
        "change_pct_vs_prev_close": round((last_close / prev_close - 1) * 100, 2),
        "hv_20d_annualised_pct": round(hv20, 2),
        "sma_20": round(float(close.tail(20).mean()), 2),
        "price_vs_sma20_pct": round((last_close / float(close.tail(20).mean()) - 1) * 100, 2),
        "last_bar_date": str(last_bar_date),
        "data_age_days": age_days,
        "today_partial_bar_dropped": dropped_partial,
        "data_status": "OK" if age_days <= MAX_DATA_AGE_DAYS else "STALE",
    }


def fetch_market_data(kinds: tuple[str, ...], now: datetime) -> dict:
    out: dict[str, dict] = {}
    for name, info in INSTRUMENTS.items():
        if info["kind"] not in kinds:
            continue
        base = {
            "symbol": info["symbol"], 
            "asset_type": info["kind"], 
            "nse_lot_size": info["lot"],
            "source": "yfinance daily bars (delayed, unofficial)"
        }
        try:
            metrics = compute_metrics(_download(info["symbol"]), now)
            out[name] = {**base, **metrics}
        except Exception as e:
            # Isolate failure: only this ticker is marked UNAVAILABLE, others load normally.
            out[name] = {**base, "data_status": "UNAVAILABLE", "reason": str(e)[:100]}
    return out


# ----------------------------------------------------------------------------
# Security & Guards
# ----------------------------------------------------------------------------
def get_secret(name: str, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def check_password() -> bool:
    expected = get_secret("APP_PASSWORD", "")
    if not expected:
        st.warning("APP_PASSWORD is not set. Anyone with this URL can spend your API credit.")
        return True
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), str(expected).encode()):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


@st.cache_resource
def _call_counter() -> dict:
    return {}


def allow_call(limit: int) -> bool:
    counter = _call_counter()
    key = now_ist().strftime("%Y-%m-%d")
    if counter.get(key, 0) >= limit:
        return False
    counter[key] = counter.get(key, 0) + 1
    return True


# ----------------------------------------------------------------------------
# Macro Research (Anthropic Web Search Tool)
# ----------------------------------------------------------------------------
RESEARCH_DOMAINS = [
    "business-standard.com", "kotakneo.com", "hdfcsky.com", "niftytrader.in",
    "quantsapp.com", "tapetide.com", "tradingeconomics.com", "reuters.com",
    "cnbc.com", "investing.com", "economictimes.indiatimes.com", "livemint.com",
]

RESEARCH_SYSTEM = """You are a market data researcher for an Indian options desk. You only collect facts with web search; you never trade and never give opinions.
For EVERY item below, report: value, source URL, the TIMESTAMP the source itself states (or NOT STATED), and a status.
Status: VERIFIED only if two independent sources agree within a small margin and the timestamp is stated; otherwise UNVERIFIED. If you cannot find it, write UNKNOWN.
Never use memory for any price or level. If sources conflict, show both values and say they conflict. Ignore pages that are old or undated for a 'latest' item.
Items: 1) GIFT Nifty level and change, 2) latest US close for Dow, S&P 500, Nasdaq, and US futures, 3) Brent crude, 4) USD/INR, 5) US 10-year yield,
6) India VIX, 7) FII and DII cash net for the latest published session (give the session date), 8) Nifty 50 and Bank Nifty latest close and session date,
9) scheduled events or major news for the next two sessions (RBI, results, geopolitics, expiries).
Output plain text, one item per line, then a short list of any conflicts. No advice."""


@st.cache_data(ttl=300, show_spinner=False)
def research_macro(_api_key: str, time_bucket: str) -> dict:
    client = anthropic.Anthropic(api_key=_api_key, timeout=120.0, max_retries=2)
    tool = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 10,
        "allowed_domains": RESEARCH_DOMAINS,
        "user_location": {"type": "approximate", "country": "IN", "timezone": "Asia/Kolkata"},
    }
    messages = [{"role": "user", "content": f"Current time: {time_bucket} IST. Collect the items now."}]
    texts: list[str] = []
    urls: list[str] = []
    searches = 0
    for _ in range(4):
        msg = client.messages.create(model=MODEL, max_tokens=4000, system=RESEARCH_SYSTEM,
                                     tools=[tool], messages=messages)
        for b in msg.content:
            t = getattr(b, "type", None)
            if t == "text":
                texts.append(b.text)
                for c in (getattr(b, "citations", None) or []):
                    if getattr(c, "url", None):
                        urls.append(c.url)
            elif t == "server_tool_use":
                searches += 1
        if msg.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": msg.content})
            continue
        break
    notes = "\n".join(texts).strip()
    if not notes:
        raise ValueError("web research returned no text")
    return {"notes": notes, "sources": list(dict.fromkeys(urls)), "searches": searches}


# ----------------------------------------------------------------------------
# AI Call & Rule Enforcement
# ----------------------------------------------------------------------------
def call_claude(api_key: str, payload: dict) -> OptionsMasterReport:
    client = anthropic.Anthropic(api_key=api_key, timeout=90.0, max_retries=2)
    tool = {
        "name": "submit_report",
        "description": "Submit the structured options desk report.",
        "input_schema": OptionsMasterReport.model_json_schema(),
    }
    msg = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_report"},
        messages=[{"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False)}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_report":
            return OptionsMasterReport.model_validate(block.input)
    raise ValueError("The model did not return the structured report.")


def enforce_rules(report: OptionsMasterReport, market_data: dict, gate: str,
                  chain_provided: bool) -> list[dict]:
    rows = []
    for rec in report.trade_recommendations:
        md = market_data.get(rec.instrument_name)
        lot = md["nse_lot_size"] if md else None
        data_status = md["data_status"] if md else "UNAVAILABLE"
        verdict = rec.verdict
        flags: list[str] = []

        if md is None:
            flags.append("Instrument name not recognised")
        if verdict == "TAKE_TRADE":
            if data_status != "OK":
                flags.append(f"Data status is {data_status}")
            if gate == "CLOSED":
                flags.append("Before the 09:30 IST candle close")
            if not chain_provided:
                flags.append("No live option-chain input was provided")
            if rec.invalidation_level is None:
                flags.append("No invalidation level given")
            if flags:
                verdict = "STAY_FLAT"
        rows.append({
            "instrument": rec.instrument_name,
            "asset_type": rec.asset_type,
            "lot_size_UNVERIFIED": lot,
            "strategy": rec.recommended_strategy,
            "outlook": rec.market_outlook,
            "entry_condition": rec.entry_condition,
            "invalidation_level": rec.invalidation_level,
            "illustrative_strikes": rec.strikes_to_deploy,
            "conditional_idea": rec.conditional_idea,
            "risk_details": rec.risk_assessment_details,
            "confidence": rec.confidence,
            "ai_verdict": rec.verdict,
            "final_verdict": verdict,
            "data_status": data_status,
            "rule_overrides": "; ".join(flags),
        })
    return rows


# ----------------------------------------------------------------------------
# Risk Calculator
# ----------------------------------------------------------------------------
def spread_risk(kind: str, width_pts: float, premium_pts: float, lot: int) -> dict:
    if kind == "Credit spread (you receive premium)":
        max_profit, max_loss = premium_pts * lot, (width_pts - premium_pts) * lot
    else:
        max_profit, max_loss = (width_pts - premium_pts) * lot, premium_pts * lot
    return {"max_profit_per_lot": round(max_profit, 2), "max_loss_per_lot": round(max_loss, 2)}


# ----------------------------------------------------------------------------
# UI Rendering
# ----------------------------------------------------------------------------
def render_results(res: dict) -> None:
    st.caption(f"Generated {res['as_of']} | session: {res['session']} | ideas gate: {res['gate']} | model: {MODEL}")
    flat_all = all(r["final_verdict"] == "STAY_FLAT" for r in res["rows"])
    if flat_all:
        st.error("OVERALL VERDICT: STAY FLAT")
    else:
        st.warning("OVERALL: at least one conditional idea. Read every override and invalidation level before acting.")

    st.subheader("Macro context")
    st.info(res["report"]["macro_market_summary"])
    if res["report"]["unknowns"]:
        st.markdown("**Unknown / not provided:** " + "; ".join(res["report"]["unknowns"]))

    st.subheader("Instrument cards")
    for r in res["rows"]:
        badge = "TAKE TRADE (conditional)" if r["final_verdict"] == "TAKE_TRADE" else "STAY FLAT"
        with st.expander(f"[{r['asset_type']}] {r['instrument']} | {badge}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Verdict", r["final_verdict"])
            c2.metric("Confidence", r["confidence"])
            c3.metric("Data", r["data_status"])
            c4.metric("Lot (unverified)", r["lot_size_UNVERIFIED"] or "n/a")
            if r["rule_overrides"]:
                st.warning(f"Downgraded to STAY_FLAT: {r['rule_overrides']}")
            st.markdown(f"**Outlook:** {r['outlook']}")
            st.markdown(f"**Strategy idea:** {r['strategy']} (strikes are illustrative: {r['illustrative_strikes']})")
            st.markdown(f"**Entry condition:** {r['entry_condition']}")
            st.markdown(f"**Invalidation level:** {r['invalidation_level']}")
            st.markdown(f"**What would make it valid:** {r['conditional_idea']}")
            st.markdown(f"**Risk:** {r['risk_details']}")

    df = pd.DataFrame(res["rows"])
    df.insert(0, "generated_ist", res["as_of"])
    df.insert(1, "session", res["session"])
    df["model"] = MODEL
    df["user_thesis"] = res["thesis"]
    df["web_sources"] = " | ".join(res.get("research", {}).get("sources", []))
    df["lot_size_source"] = LOT_SIZE_SOURCE

    st.subheader("Quick reference")
    st.dataframe(df[["instrument", "final_verdict", "confidence", "data_status",
                       "strategy", "invalidation_level", "rule_overrides"]], use_container_width=True)
    st.download_button(
        "Download CSV log",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"options_desk_log_{res['as_of'][:10]}.csv",
        mime="text/csv",
    )

    research = res.get("research", {})
    with st.expander(f"Web research Claude fetched ({research.get('searches', 0)} searches) - check the sources"):
        st.text(research.get("notes", "none"))
        for u in research.get("sources", []):
            st.markdown(f"- {u}")
    with st.expander("Price data sent to the AI (audit)"):
        st.json(res["market_data"])


def main() -> None:
    st.set_page_config(page_title="Options Quant Desk", page_icon="📊", layout="wide")
    st.title("Indian Index & Stock Options Quant Desk")
    st.caption("Read-only analysis. It places no trades and is not investment advice. You decide.")

    if not check_password():
        st.stop()

    api_key = get_secret("ANTHROPIC_API_KEY", "")
    try:
        limit = int(get_secret("DAILY_CALL_LIMIT", 20))
    except (TypeError, ValueError):
        limit = 20

    now = now_ist()
    label, gate = session_label(now), ideas_gate(now)
    st.sidebar.markdown(f"**IST now:** {now:%a %d %b %H:%M}  \n**Session:** {label}  \n**Ideas gate:** {gate}")
    st.sidebar.caption("NSE holidays are not checked.")
    if gate == "CLOSED":
        st.sidebar.warning("Before the 09:30 IST candle close, every verdict is forced to STAY_FLAT.")

    st.sidebar.header("1. Segment")
    mode = st.sidebar.selectbox("Market segment", list(MODES))

    st.sidebar.header("2. Macro baseline")
    auto_macro = st.sidebar.checkbox(
        "Let Claude fetch macro data from the web when I press Run", value=True,
        help="Uses the Anthropic web search tool (about $0.10 and 30-90 seconds per run).")
    st.sidebar.caption("Optional overrides: anything you type here beats the web value. Blank = no override.")
    macro_fields = ["GIFT Nifty", "Brent crude ($)", "USD/INR", "US 10Y yield (%)",
                    "India VIX", "FII cash net (Rs Cr)", "DII cash net (Rs Cr)", "Key news / events"]
    macro = {f: (st.sidebar.text_input(f).strip() or "UNKNOWN") for f in macro_fields}
    macro_source = st.sidebar.text_input("Macro source and time (e.g. 'Groww 09:40 IST')").strip() or "NOT GIVEN"

    st.sidebar.header("3. Option chain (you paste it)")
    chain_text = st.sidebar.text_area(
        "Expiry, ATM IV, straddle, max call/put OI strikes, premiums",
        height=140).strip()
    thesis = st.sidebar.text_area("Your thesis / notes (saved in the CSV)", height=80).strip()

    run = st.sidebar.button("Run desk audit", type="primary")

    with st.expander("Risk calculator (per lot, uses premiums you read from the live chain)"):
        k = st.selectbox("Structure", ["Credit spread (you receive premium)",
                                       "Debit spread (you pay premium)"])
        names = list(INSTRUMENTS)
        pick = st.selectbox("Instrument", names)
        default_lot = INSTRUMENTS[pick]["lot"]
        a, b, c = st.columns(3)
        width = a.number_input("Strike width (points)", min_value=0.0, value=100.0, step=50.0)
        prem = b.number_input("Net premium (points)", min_value=0.0, value=0.0, step=0.5)
        lot = c.number_input("Lot size (verify on NSE)", min_value=1, value=int(default_lot), step=1)
        if prem >= width > 0:
            st.error("Premium must be less than the strike width.")
        elif width > 0 and prem > 0:
            r = spread_risk(k, width, prem, int(lot))
            st.write(f"Max profit per lot: Rs {r['max_profit_per_lot']:,}  |  "
                     f"Max loss per lot: Rs {r['max_loss_per_lot']:,}")

    if run:
        if not api_key:
            st.error("Add ANTHROPIC_API_KEY to Streamlit Secrets.")
        elif not allow_call(limit):
            st.error(f"Daily call limit ({limit}) reached.")
        else:
            with st.spinner("Fetching data and running the audit..."):
                try:
                    market_data = fetch_market_data(MODES[mode], now)
                    n_ok = sum(1 for v in market_data.values() if v.get("data_status") == "OK")
                    if n_ok == 0:
                        st.error("No instrument has fresh data. Verdict: STAY FLAT. Try again later.")
                        st.stop()
                    research = {"notes": "NOT REQUESTED", "sources": [], "searches": 0}
                    if auto_macro:
                        try:
                            bucket = f"{now.replace(minute=now.minute // 10 * 10):%Y-%m-%d %H:%M}"
                            research = research_macro(api_key, bucket)
                        except Exception as e:
                            research = {"notes": f"WEB RESEARCH FAILED: {str(e)[:150]}. Treat all macro as UNKNOWN.",
                                        "sources": [], "searches": 0}
                            st.warning("Web research failed, so macro is UNKNOWN for this run.")
                    overrides = {k: v for k, v in macro.items() if v != "UNKNOWN"}
                    payload = {
                        "as_of_ist": f"{now:%Y-%m-%d %H:%M}",
                        "session_state": label,
                        "ideas_gate": gate,
                        "mode": mode,
                        "market_data": market_data,
                        "live_web_research": research,
                        "user_macro_overrides_unverified": {**overrides, "source_and_time": macro_source} if overrides else "NONE",
                        "option_chain_input": chain_text or "NOT PROVIDED",
                    }
                    report = call_claude(api_key, payload)
                    rows = enforce_rules(report, market_data, gate, bool(chain_text))
                    st.session_state["result"] = {
                        "as_of": f"{now:%Y-%m-%d %H:%M IST}", "session": label, "gate": gate,
                        "report": report.model_dump(), "rows": rows,
                        "market_data": market_data, "thesis": thesis, "research": research,
                    }
                except ValidationError as e:
                    st.error(f"The AI reply failed validation, so there is no result. Verdict: STAY FLAT.\n\n{e}")
                except Exception as e:
                    st.error(f"Run failed. Verdict: STAY FLAT.\n\n{e}")

    if "result" in st.session_state:
        render_results(st.session_state["result"])
    else:
        st.info("Set the inputs in the sidebar and press Run desk audit. Default answer: STAY FLAT unless the evidence is clear.")


if __name__ == "__main__":
    main()
