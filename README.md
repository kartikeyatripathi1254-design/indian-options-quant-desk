# Indian Options Quant Desk

[![CI](https://github.com/kartikeyatripathi1254-design/indian-options-quant-desk/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikeyatripathi1254-design/indian-options-quant-desk/actions/workflows/ci.yml)

A read-only research assistant for options on Indian indices (Nifty 50, Bank Nifty, Sensex) and liquid NSE stocks.
It pulls daily price data, computes realised volatility and trend metrics, collects the latest macro context with
Claude's web search, and asks Claude for a structured view per instrument. A deterministic rule layer then
downgrades any idea that isn't backed by fresh data, a confirmed session window, a live option chain and an
invalidation level.

> **Not investment advice.** The app places no trades, never asks for broker logins, and defaults to **STAY FLAT**.

## How it works

1. **Market data** – `yfinance` daily closes per instrument, fetched and cached independently, so one failed
   ticker doesn't break the run. Today's incomplete bar is dropped during market hours; data older than 4 days is
   marked `STALE`.
2. **Metrics** – 20-day annualised historical volatility (HV-20, from log returns), last close, change vs the
   previous close, 20-day SMA and distance from it.
3. **Macro research** – Claude with the Anthropic web-search tool, restricted to a list of financial news domains.
   Every item must carry a source URL and the source's own timestamp, and is labelled VERIFIED only when two
   sources agree.
4. **Structured AI output** – Claude must answer through a tool whose input schema is generated from Pydantic
   models (`OptionsMasterReport` / `Recommendation`); the reply is validated before use. The system prompt forbids
   invented prices, win rates and probabilities of profit, and treats HV-20 as realised (not implied) volatility.
5. **Rule enforcement** – `enforce_rules()` overrides the AI: a `TAKE_TRADE` becomes `STAY_FLAT` if the data is
   stale or unavailable, it is before the 09:30 IST candle close, no option chain was pasted, the instrument name
   is unknown, or no invalidation level was given. Every override is shown to the user.
6. **Risk calculator** – max profit and max loss per lot for credit and debit spreads, from premiums read off the
   live chain.
7. **Audit trail** – a CSV log with the verdicts, overrides, user thesis, web sources and the exact data sent
   to the model.

Guard rails for a public deployment: an optional app password (constant-time comparison) and a daily limit on
paid API calls.

## Tech stack

Python 3.10+ · Streamlit · Anthropic Claude API (tool use + web search) · Pydantic v2 · yfinance · pandas · NumPy ·
pytest · Docker · GitHub Actions

## Run locally

```bash
git clone https://github.com/kartikeyatripathi1254-design/indian-options-quant-desk.git
cd indian-options-quant-desk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your Anthropic API key
streamlit run app.py
```

### Run with Docker

```bash
docker build -t options-desk .
docker run -p 8501:8501 -v "$PWD/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" options-desk
```

### Deploy on Streamlit Community Cloud

Point a new app at `app.py` and paste the contents of `secrets.toml.example` (with real values) into
**App settings → Secrets**. Set `APP_PASSWORD` so strangers can't spend your API credit.

## Tests

```bash
python -m pytest -q
```

The tests cover the session and ideas-gate logic, the HV-20 calculation (checked against a manual computation),
partial-bar and stale-data handling, every rule in `enforce_rules()`, and the spread risk calculator. They need
no network access or API key. CI runs linting, the tests and a Docker build on every push.

## Limitations

- yfinance data is delayed and unofficial; NSE holidays are not checked.
- Lot sizes are hard-coded and must be checked against the exchange before use.
- The app has no implied-volatility or option-chain feed of its own; the user pastes chain data.

## License

MIT
