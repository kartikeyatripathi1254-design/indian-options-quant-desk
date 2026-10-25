# indian-options-quant-desk
Institutional-grade quant options desk for Indian indices &amp; stocks. Built with Python, yfinance, Streamlit, and Gemini AI for automated volatility modeling, structured trade setups, win-rate probabilities, and risk management.
# 📊 Indian Index & Stock Options Quant Desk

An institutional-grade quantitative options trading desk and screener tailored for the Indian financial markets (Nifty 50, Sensex, Nifty Midcap, and high-liquidity NSE stocks). Built with Python, `yfinance`, Pydantic, Streamlit, and Google AI Studio (Gemini 3.8 Flash).

## 🚀 Key Features
* **Multi-Segment Scans:** Analyze Nifty 50, Sensex, Midcap, and top NSE equities (Reliance, TCS, HDFC Bank, Infosys, Tata Motors).
* **Quantitative Volatility Modeling:** Calculates 20-day annualized historical volatility via log returns and Close-to-Close metrics.
* **Structured AI Intelligence:** Leverages Gemini 3.8 Flash with enforced Pydantic JSON schemas to deliver institutional-grade trade setups.
* **Risk & Probability Metrics:** Outputs estimated win-rate probabilities, max reward/risk per lot, exact strike selections, and dynamic risk management briefs.
* **Export Functionality:** Instantly download your options strategy logs as a structured CSV report.
* **Mobile-Accessible:** Easily hosted via Streamlit Community Cloud for on-the-go monitoring.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit
* **Financial Data:** `yfinance`, `pandas`, `numpy`
* **Data Validation:** `pydantic`
* **AI Engine:** Google GenAI SDK (`google-genai` / Gemini 3.8 Flash)

---

## 💻 Local Setup & Running

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
