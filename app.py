"""
Nexus AI — SaaS Financial Intelligence Dashboard
==================================================
A driver-based FP&A model for a fictional B2B AI SaaS company, built to
demonstrate: SaaS metrics engineering, scenario-based forecasting, and
AI-assisted board reporting.

Author: Kartikeya Tripathi (FP&A Manager / Finance Business Partner)
Portfolio project — not real company data.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Optional: paste an Anthropic API key in the sidebar to generate CFO
commentary with Claude instead of the built-in rules-based narrative
engine. The key is only ever held in this Python process (server-side),
never exposed to the browser — unlike a client-side JS implementation.
"""

import io
import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    import requests
except ImportError:
    requests = None

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ======================================================================
# THEME
# ======================================================================
TEAL = "#0D7377"
TEAL_LIGHT = "#14FFEC"
DARK = "#2D2D2D"
GRAY = "#8A8A8A"
BG = "#FAFAFA"

st.set_page_config(
    page_title="Nexus AI — FP&A Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates["nexus"] = go.layout.Template(
    layout=go.Layout(
        colorway=[TEAL, "#F2A65A", DARK, "#5FB8B6", "#C94C4C", GRAY],
        font=dict(color=DARK, family="Helvetica, Arial, sans-serif", size=13),
        title_font=dict(size=17, color=DARK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#EDEDED", zeroline=False),
        yaxis=dict(gridcolor="#EDEDED", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=10, r=10),
    )
)
pio.templates.default = "nexus"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG}; }}
        h1, h2, h3 {{ color: {TEAL}; font-family: Helvetica, Arial, sans-serif; }}
        [data-testid="stMetricValue"] {{ color: {TEAL}; font-weight: 700; }}
        [data-testid="stMetricLabel"] {{ color: {DARK}; font-weight: 500; }}
        [data-testid="stMetricDelta"] svg {{ display: none; }}
        .stButton>button {{
            background-color: {TEAL}; color: white; border: none; border-radius: 6px;
            font-weight: 600; padding: 0.5rem 1.2rem;
        }}
        .stButton>button:hover {{ background-color: {TEAL_LIGHT}; color: {DARK}; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {DARK}; }}
        .stTabs [aria-selected="true"] {{ color: {TEAL} !important; }}
        .commentary-box {{
            background-color: white; border-left: 4px solid {TEAL};
            padding: 1.2rem 1.5rem; border-radius: 4px; line-height: 1.6;
        }}
        .footer-note {{ color: {GRAY}; font-size: 0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# SCENARIO PRESETS
# Net MoM growth target, gross churn, expansion, ARPU, CAC, gross margin
# ======================================================================
SCENARIOS = {
    "Base": dict(growth_target=0.08, churn=0.025, expansion=0.04, arpu=3000,
                 cac=12000, gross_margin=0.78, headcount_cost=300000,
                 other_opex=110000, opex_growth=0.015),
    "Bull": dict(growth_target=0.12, churn=0.018, expansion=0.055, arpu=3200,
                 cac=11000, gross_margin=0.80, headcount_cost=300000,
                 other_opex=110000, opex_growth=0.02),
    "Bear": dict(growth_target=0.04, churn=0.040, expansion=0.020, arpu=2800,
                 cac=14000, gross_margin=0.74, headcount_cost=300000,
                 other_opex=110000, opex_growth=0.01),
}

SCENARIO_BLURB = {
    "Base": "Management case — current trajectory continues.",
    "Bull": "Upside case — stronger new-logo growth, lower churn, faster expansion.",
    "Bear": "Downside case — slower growth, elevated churn, CAC inflation.",
}

# Formula reference — used both for the in-app methodology tab and for
# keeping the model auditable.
FORMULAS = {
    "ARR": "Ending MRR × 12",
    "MRR Growth % (MoM)": "(Ending MRR − Beginning MRR) ÷ Beginning MRR",
    "Net Dollar Retention (NDR)": "(Beginning MRR + Expansion MRR − Churned MRR) ÷ Beginning MRR "
                                   "— measured on the existing customer cohort, excludes new logos",
    "Gross Margin %": "(Revenue − COGS) ÷ Revenue",
    "CAC (per customer)": "Total CAC Spend ÷ New Customers Acquired",
    "LTV": "(ARPU × Gross Margin %) ÷ Monthly Gross Churn Rate",
    "LTV : CAC Ratio": "LTV ÷ CAC",
    "CAC Payback (months)": "CAC ÷ (ARPU × Gross Margin %)",
    "Net Burn": "Total Costs (COGS + Opex) − Revenue",
    "Cash Runway (months)": "Cash on Hand ÷ Current Monthly Net Burn",
    "Burn Multiple": "Net Burn ÷ Net New ARR  (Bessemer efficiency metric; lower is better, <1x is excellent)",
    "Rule of 40": "Annualized MRR Growth Rate (%) + EBITDA Margin (%)  — ≥40% is considered healthy for SaaS",
}


# ======================================================================
# CORE MODEL
# ======================================================================
@st.cache_data(show_spinner=False)
def build_monthly_model(start_mrr, start_cash, rd_pct, growth_target, churn,
                         expansion, arpu, cac, gross_margin, headcount_cost,
                         other_opex, opex_growth, months=12):
    """
    Builds a rolling monthly driver-based P&L and SaaS metrics forecast.

    The model is fully deterministic and forward-calculated from the input
    drivers below — nothing is hardcoded except the starting balance sheet
    position (starting MRR / starting cash).

    Cost allocation convention:
      - Headcount cost splits R&D / S&M / G&A using `rd_pct` for R&D, and
        a fixed 62.5% / 37.5% split of the remainder to S&M / G&A
        (a reasonable early-stage SaaS blend).
      - CAC spend (variable, driven by new customers acquired) is added
        to S&M. "Other Opex" (tools, facilities, non-headcount G&A) is
        added to G&A.
      - COGS = Revenue × (1 − Gross Margin %).
    """
    rows = []
    beginning_mrr = start_mrr
    cash_begin = start_cash

    for m in range(1, months + 1):
        new_mrr = beginning_mrr * max(growth_target - expansion + churn, 0.0)
        expansion_mrr = beginning_mrr * expansion
        churned_mrr = beginning_mrr * churn
        ending_mrr = beginning_mrr + new_mrr + expansion_mrr - churned_mrr

        new_customers = new_mrr / arpu if arpu else 0.0
        churned_customers = churned_mrr / arpu if arpu else 0.0

        revenue = ending_mrr
        cogs = revenue * (1 - gross_margin)
        gross_profit = revenue - cogs

        hc_cost_m = headcount_cost * ((1 + opex_growth) ** (m - 1))
        other_opex_m = other_opex * ((1 + opex_growth) ** (m - 1))
        cac_spend = new_customers * cac

        rd_cost = hc_cost_m * rd_pct
        sm_cost = hc_cost_m * 0.625 * (1 - rd_pct) + cac_spend
        ga_cost = hc_cost_m * 0.375 * (1 - rd_pct) + other_opex_m

        total_opex = rd_cost + sm_cost + ga_cost
        total_costs = cogs + total_opex
        ebitda = revenue - total_costs
        net_burn = -ebitda

        cash_end = cash_begin + ebitda
        runway_months = (cash_end / net_burn) if net_burn > 0 else np.inf

        ndr = ((beginning_mrr + expansion_mrr - churned_mrr) / beginning_mrr
               if beginning_mrr else np.nan)
        arr = ending_mrr * 12
        mrr_growth = (ending_mrr - beginning_mrr) / beginning_mrr if beginning_mrr else np.nan
        ltv = (arpu * gross_margin) / max(churn, 0.0001)
        cac_payback_months = cac / (arpu * gross_margin) if (arpu * gross_margin) else np.nan
        ltv_cac = ltv / cac if cac else np.nan
        net_new_arr = (ending_mrr - beginning_mrr) * 12
        burn_multiple = (net_burn / net_new_arr) if net_new_arr > 0 else np.nan
        ebitda_margin = ebitda / revenue if revenue else np.nan
        rule_of_40 = (mrr_growth * 12 * 100) + (ebitda_margin * 100)

        rows.append(dict(
            month=m, beginning_mrr=beginning_mrr, new_mrr=new_mrr,
            expansion_mrr=expansion_mrr, churned_mrr=churned_mrr, ending_mrr=ending_mrr,
            new_customers=new_customers, churned_customers=churned_customers,
            revenue=revenue, cogs=cogs, gross_profit=gross_profit,
            rd_cost=rd_cost, sm_cost=sm_cost, ga_cost=ga_cost, cac_spend=cac_spend,
            total_opex=total_opex, total_costs=total_costs, ebitda=ebitda,
            ebitda_margin=ebitda_margin, net_burn=net_burn, cash_end=cash_end,
            runway_months=runway_months, ndr=ndr, arr=arr, mrr_growth=mrr_growth,
            ltv=ltv, cac_payback_months=cac_payback_months, ltv_cac=ltv_cac,
            burn_multiple=burn_multiple, gross_margin_pct=gross_profit / revenue if revenue else np.nan,
            rule_of_40=rule_of_40,
        ))

        beginning_mrr = ending_mrr
        cash_begin = cash_end

    return pd.DataFrame(rows)


# ======================================================================
# FORMATTING HELPERS
# ======================================================================
def fmt_eur(x, decimals=0):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:,.{max(decimals,1)}f}M"
    if abs(x) >= 1_000:
        return f"€{x/1_000:,.0f}K"
    return f"€{x:,.{decimals}f}"


def fmt_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x*100:,.{decimals}f}%"


def fmt_x(x, decimals=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:,.{decimals}f}x"


def fmt_months(x, decimals=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "∞"
    if x < 0:
        return "0 (cash-out)"
    return f"{x:,.{decimals}f} mo"


# ======================================================================
# CFO COMMENTARY — RULES ENGINE (fallback / default)
# ======================================================================
def generate_template_commentary(df, meta):
    first, last = df.iloc[0], df.iloc[-1]
    company = meta["company_name"]
    scenario = meta["scenario"]

    arr_start = meta["start_mrr"] * 12
    arr_end = last["arr"]
    arr_growth_pct = (arr_end - arr_start) / arr_start * 100 if arr_start else 0
    avg_ndr = df["ndr"].mean() * 100
    avg_burn_multiple = df["burn_multiple"].replace([np.inf, -np.inf], np.nan).mean()

    cash_out_row = df[df["cash_end"] < 0]
    cash_out_month = int(cash_out_row.iloc[0]["month"]) if not cash_out_row.empty else None

    # ---- Paragraph 1: Highlights ----
    p1 = (
        f"{company} closed the trailing 12-month {scenario.lower()}-case forecast at "
        f"{fmt_eur(arr_end)} ARR, up {arr_growth_pct:.0f}% from a {fmt_eur(arr_start)} starting base, "
        f"driven by {fmt_eur(last['new_mrr'])} in new business and {fmt_eur(last['expansion_mrr'])} "
        f"in expansion revenue in the final month alone. Net Dollar Retention averaged "
        f"{avg_ndr:.0f}% across the period"
    )
    if avg_ndr >= 100:
        p1 += ", indicating the existing customer base is expanding faster than it churns — a durable growth engine independent of new-logo acquisition. "
    else:
        p1 += ", meaning the existing book is contracting net of expansion and must be offset by new-logo growth to sustain top-line momentum. "
    p1 += (
        f"Unit economics remain a bright spot: LTV:CAC stands at {fmt_x(last['ltv_cac'])} "
        f"against a {fmt_x(4)} underwriting target, with CAC payback of "
        f"{last['cac_payback_months']:.1f} months — well inside the 12-18 month range typical for efficient B2B SaaS."
    )

    # ---- Paragraph 2: Risks ----
    risk_clauses = []
    if cash_out_month is not None:
        risk_clauses.append(
            f"at the current burn trajectory, cash on hand is projected to be exhausted by month {cash_out_month} "
            f"of the forecast window — this is the single most time-sensitive item for the board"
        )
    else:
        risk_clauses.append(
            f"cash runway at the end of the forecast period stands at {fmt_months(last['runway_months'])}, "
            f"which {'comfortably clears' if (isinstance(last['runway_months'], float) and last['runway_months'] >= 18) else 'sits inside'} "
            f"the 18-month threshold typically expected ahead of a next funding round"
        )
    if avg_ndr < 100:
        risk_clauses.append("net retention below 100% signals churn or downgrade pressure that should be root-caused with Customer Success before it compounds")
    if last["burn_multiple"] > 1.5:
        risk_clauses.append(f"the burn multiple of {fmt_x(last['burn_multiple'])} is elevated — more than {fmt_x(last['burn_multiple'])} of cash is being spent per euro of net-new ARR, which will draw scrutiny from growth-stage investors")
    if last["cac_payback_months"] > 18:
        risk_clauses.append("CAC payback has drifted beyond the 18-month efficiency threshold, implying paid acquisition is getting less capital-efficient")
    if not risk_clauses:
        risk_clauses.append("no metric in this forecast currently breaches its efficiency threshold, though the model should be re-run monthly as actuals land, since small drift in churn or CAC compounds quickly over a 12-month horizon")

    p2 = "On risk: " + "; ".join(risk_clauses) + "."

    # ---- Paragraph 3: Recommendations ----
    recs = []
    if cash_out_month is not None or (isinstance(last["runway_months"], float) and last["runway_months"] < 12):
        recs.append("initiate fundraising conversations now, targeting a close with at least 6 months of buffer before the projected cash-out point")
        recs.append("run a parallel opex review to identify 10-15% of non-critical spend (tooling, contractor, discretionary G&A) that can be deferred without slowing the roadmap")
    if avg_ndr < 100:
        recs.append("stand up a proactive Customer Success motion for at-risk accounts (usage-based health scoring, QBRs for top-quartile accounts) to convert churn risk into expansion")
    if last["burn_multiple"] > 1.2:
        recs.append("tighten CAC efficiency by reallocating spend toward the highest-converting channel identified in the CAC-by-channel breakdown, and hold headcount growth flat for one quarter")
    recs.append(f"continue monitoring LTV:CAC and NDR monthly — both are currently ahead of plan and represent the strongest argument in the next fundraising narrative")

    p3 = "Recommended actions: " + "; ".join(recs) + "."

    return f"{p1}\n\n{p2}\n\n{p3}"


def generate_ai_commentary(df, meta, api_key):
    """Calls the Anthropic API to generate commentary using the same
    computed KPIs as the template engine, for a more natural narrative.
    Falls back to the template engine on any failure."""
    if requests is None or not api_key:
        return generate_template_commentary(df, meta), "template"

    last = df.iloc[-1]
    kpi_context = {
        "company": meta["company_name"],
        "scenario": meta["scenario"],
        "arr_end": round(last["arr"]),
        "arr_start": round(meta["start_mrr"] * 12),
        "avg_ndr_pct": round(df["ndr"].mean() * 100, 1),
        "ltv_cac": round(last["ltv_cac"], 1),
        "cac_payback_months": round(last["cac_payback_months"], 1),
        "net_burn_latest": round(last["net_burn"]),
        "cash_end_latest": round(last["cash_end"]),
        "runway_months_latest": (
            round(last["runway_months"], 1) if np.isfinite(last["runway_months"]) else "infinite (cash flow positive)"
        ),
        "burn_multiple_latest": round(last["burn_multiple"], 2) if np.isfinite(last["burn_multiple"]) else None,
        "gross_margin_pct": round(last["gross_margin_pct"] * 100, 1),
    }

    system_prompt = (
        "You are a CFO writing board-pack commentary. Write exactly three paragraphs, "
        "no headers, no bullet points: (1) highlights, (2) risks, (3) recommendations. "
        "Be direct, quantitative, and concise — the tone of an experienced FP&A leader, "
        "not a marketing summary. Use the exact figures given; do not invent numbers."
    )
    user_prompt = f"Here is this month's SaaS KPI snapshot:\n{json.dumps(kpi_context, indent=2)}\n\nWrite the board commentary."

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 700,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        text = "\n\n".join(text_blocks).strip()
        if text:
            return text, "claude"
        return generate_template_commentary(df, meta), "template"
    except Exception:
        return generate_template_commentary(df, meta), "template"


# ======================================================================
# PDF EXPORT
# ======================================================================
def pdf_safe(text):
    """fpdf's core Helvetica font is Latin-1 only. Transliterate common
    'smart' punctuation and the euro sign, then use a safe fallback for
    anything else (e.g. unexpected characters from an LLM-generated
    commentary) so PDF export never crashes."""
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u20ac": "EUR ",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_report(company_name, scenario, kpis, commentary):
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(13, 115, 119)
    pdf.cell(0, 10, pdf_safe(f"{company_name} - Board Financial Summary"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, pdf_safe(f"Scenario: {scenario}  |  Generated: {date.today().isoformat()}"),
              new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_draw_color(13, 115, 119)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(45, 45, 45)
    pdf.cell(0, 8, "Key Metrics (Month 12)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for k, v in kpis.items():
        pdf.set_text_color(45, 45, 45)
        pdf.cell(70, 7, pdf_safe(k), new_x="RIGHT", new_y="TOP")
        pdf.set_text_color(13, 115, 119)
        pdf.cell(0, 7, pdf_safe(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(45, 45, 45)
    pdf.cell(0, 8, "CFO Commentary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(45, 45, 45)
    for para in commentary.split("\n\n"):
        pdf.multi_cell(0, 6, pdf_safe(para))
        pdf.ln(2)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.ln(4)
    pdf.multi_cell(0, 5, pdf_safe("Fictional company data for portfolio demonstration purposes. "
                          "Built by Kartikeya Tripathi -- FP&A Manager / Finance Business Partner."))

    out = pdf.output()
    return bytes(out)


# ======================================================================
# SIDEBAR — INPUTS
# ======================================================================
with st.sidebar:
    st.markdown(f"### 📊 Model Inputs")
    company_name = st.text_input("Company name", value="Nexus AI")

    st.markdown("**Starting position (today)**")
    start_mrr = st.number_input("Starting MRR (€)", min_value=0, value=180_000, step=5_000)
    start_cash = st.number_input("Cash on hand (€)", min_value=0, value=2_400_000, step=50_000)
    headcount = st.number_input("Headcount (FTEs)", min_value=1, value=18, step=1)
    rd_pct = st.slider("R&D share of headcount cost", 0, 100, 60, help="Remaining % splits 62.5/37.5 between S&M / G&A") / 100

    st.markdown("---")
    st.markdown("**Scenario**")
    scenario = st.radio("Select scenario", list(SCENARIOS.keys()), horizontal=True,
                          format_func=lambda s: s)
    st.caption(SCENARIO_BLURB[scenario])

    preset = SCENARIOS[scenario]
    st.markdown("**Growth & retention drivers**")
    growth_target = st.slider("Target net MoM MRR growth", 0.0, 20.0,
                                preset["growth_target"] * 100, 0.5,
                                key=f"growth_{scenario}", format="%.1f%%") / 100
    churn = st.slider("Gross monthly churn (% of beginning MRR)", 0.0, 10.0,
                        preset["churn"] * 100, 0.1, key=f"churn_{scenario}", format="%.1f%%") / 100
    expansion = st.slider("Monthly expansion revenue (% of beginning MRR)", 0.0, 10.0,
                            preset["expansion"] * 100, 0.1, key=f"exp_{scenario}", format="%.1f%%") / 100

    st.markdown("**Unit economics**")
    arpu = st.number_input("ARPU (€/customer/month)", min_value=1, value=preset["arpu"],
                             step=100, key=f"arpu_{scenario}")
    cac = st.number_input("CAC (€/new customer)", min_value=1, value=preset["cac"],
                            step=500, key=f"cac_{scenario}")
    gross_margin = st.slider("Gross margin %", 30, 95, int(preset["gross_margin"] * 100),
                               1, key=f"gm_{scenario}") / 100

    st.markdown("**Cost structure**")
    headcount_cost = st.number_input("Total headcount cost (€/month, fully loaded)",
                                       min_value=0, value=preset["headcount_cost"], step=10_000,
                                       key=f"hc_{scenario}")
    other_opex = st.number_input("Other opex (€/month: tools, facilities, non-HC G&A)",
                                   min_value=0, value=preset["other_opex"], step=5_000,
                                   key=f"opex_{scenario}")
    opex_growth = st.slider("Monthly opex growth (hiring/inflation)", 0.0, 5.0,
                              preset["opex_growth"] * 100, 0.1, key=f"opexg_{scenario}",
                              format="%.1f%%") / 100

    st.markdown("---")
    st.markdown("**🤖 AI Commentary (optional)**")
    api_key_input = st.text_input("Anthropic API key", type="password",
                                    help="Leave blank to use the built-in rules-based narrative engine. "
                                         "Key is used only for this session and never stored or logged.")
    st.caption("Without a key, commentary is generated by a deterministic rules "
               "engine that reads the same computed KPIs.")

    st.markdown("---")
    st.markdown(
        '<p class="footer-note">Built by Kartikeya Tripathi — FP&A Manager /<br>'
        'Finance Business Partner, Berlin<br>Portfolio project · fictional data</p>',
        unsafe_allow_html=True,
    )


# ======================================================================
# COMPUTE MODEL(S)
# ======================================================================
df = build_monthly_model(
    start_mrr=start_mrr, start_cash=start_cash, rd_pct=rd_pct,
    growth_target=growth_target, churn=churn, expansion=expansion,
    arpu=arpu, cac=cac, gross_margin=gross_margin, headcount_cost=headcount_cost,
    other_opex=other_opex, opex_growth=opex_growth,
)

# Reference scenarios (fixed presets) for the comparison chart — independent
# of any live slider edits on the active scenario, so the comparison always
# reflects the three canonical cases.
comparison_dfs = {}
for s_name, s_assump in SCENARIOS.items():
    comparison_dfs[s_name] = build_monthly_model(
        start_mrr=start_mrr, start_cash=start_cash, rd_pct=rd_pct,
        **s_assump,
    )

meta = dict(company_name=company_name, scenario=scenario, start_mrr=start_mrr)


# ======================================================================
# HEADER
# ======================================================================
st.title(f"📊 {company_name} — FP&A Intelligence Dashboard")
st.caption(
    f"Driver-based SaaS financial model · Scenario: **{scenario}** "
    f"({SCENARIO_BLURB[scenario]}) · 12-month rolling forecast"
)

last = df.iloc[-1]

tab_overview, tab_forecast, tab_unit, tab_ai, tab_formulas = st.tabs(
    ["📊 Overview", "📈 Forecast & Bridge", "🎯 Unit Economics", "🤖 CFO Commentary", "📐 Formulas & Methodology"]
)


# ----------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ----------------------------------------------------------------------
with tab_overview:
    st.markdown("#### Headline KPIs — Month 12 of forecast")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARR", fmt_eur(last["arr"]),
               f"{fmt_pct(last['mrr_growth'])} MoM")
    c2.metric("Net Dollar Retention", fmt_pct(last["ndr"]),
               "Existing-book health")
    c3.metric("Cash Runway", fmt_months(last["runway_months"]),
               f"Cash: {fmt_eur(last['cash_end'])}")
    c4.metric("Rule of 40", f"{last['rule_of_40']:.0f}%",
               "Growth % + EBITDA margin %")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Gross Margin", fmt_pct(last["gross_margin_pct"]))
    c6.metric("CAC Payback", f"{last['cac_payback_months']:.1f} mo")
    c7.metric("LTV : CAC", fmt_x(last["ltv_cac"]), "Target: 4.0x")
    c8.metric("Burn Multiple", fmt_x(last["burn_multiple"]),
               "Net burn / Net new ARR")

    if df[df["cash_end"] < 0].shape[0] > 0:
        cash_out_m = int(df[df["cash_end"] < 0].iloc[0]["month"])
        st.warning(
            f"⚠️ At current assumptions, cash on hand is projected to go negative in **month {cash_out_m}** "
            f"of the forecast. See the CFO Commentary tab for a full risk read and recommended actions."
        )

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig_mrr = go.Figure()
        fig_mrr.add_trace(go.Scatter(x=df["month"], y=df["ending_mrr"], mode="lines+markers",
                                       name="MRR", line=dict(color=TEAL, width=3)))
        fig_mrr.update_layout(title="MRR Trend — 12 Month Forecast", height=340,
                                xaxis_title="Month", yaxis_title="MRR (€)")
        st.plotly_chart(fig_mrr, width="stretch")

    with col_b:
        fig_cash = go.Figure()
        fig_cash.add_trace(go.Scatter(x=df["month"], y=df["cash_end"], mode="lines+markers",
                                        name="Cash Balance", line=dict(color=DARK, width=3),
                                        fill="tozeroy", fillcolor="rgba(13,115,119,0.08)"))
        fig_cash.add_hline(y=0, line_dash="dash", line_color="#C94C4C")
        fig_cash.update_layout(title="Cash Balance vs. Burn", height=340,
                                 xaxis_title="Month", yaxis_title="Cash (€)")
        st.plotly_chart(fig_cash, width="stretch")

    st.markdown("#### Scenario comparison — ARR trajectory")
    fig_scenarios = go.Figure()
    colors = {"Base": TEAL, "Bull": "#5FB8B6", "Bear": "#C94C4C"}
    for s_name, s_df in comparison_dfs.items():
        fig_scenarios.add_trace(go.Scatter(
            x=s_df["month"], y=s_df["arr"], mode="lines+markers", name=s_name,
            line=dict(color=colors[s_name], width=2.5, dash="solid" if s_name == "Base" else "dot"),
        ))
    fig_scenarios.update_layout(height=360, xaxis_title="Month", yaxis_title="ARR (€)",
                                  title="ARR Trajectory — Base / Bull / Bear (preset assumptions)")
    st.plotly_chart(fig_scenarios, width="stretch")
    st.caption(
        "Comparison uses each scenario's preset assumptions (sidebar sliders only affect the "
        "currently active scenario shown throughout the rest of the dashboard)."
    )


# ----------------------------------------------------------------------
# TAB 2 — FORECAST & BRIDGE
# ----------------------------------------------------------------------
with tab_forecast:
    st.markdown("#### Revenue bridge — Month 12 waterfall")
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Beginning MRR", "New Business", "Expansion", "Churn", "Ending MRR"],
        y=[last["beginning_mrr"], last["new_mrr"], last["expansion_mrr"], -last["churned_mrr"], 0],
        connector={"line": {"color": "#C0C0C0"}},
        decreasing={"marker": {"color": "#C94C4C"}},
        increasing={"marker": {"color": TEAL}},
        totals={"marker": {"color": DARK}},
        text=[fmt_eur(v) for v in [last["beginning_mrr"], last["new_mrr"], last["expansion_mrr"],
                                      -last["churned_mrr"], last["ending_mrr"]]],
        textposition="outside",
    ))
    fig_wf.update_layout(height=420, title=f"MRR Bridge — Month 12 ({scenario} case)")
    st.plotly_chart(fig_wf, width="stretch")

    st.markdown("#### Revenue composition by month")
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(x=df["month"], y=df["new_mrr"], name="New Business", marker_color=TEAL))
    fig_stack.add_trace(go.Bar(x=df["month"], y=df["expansion_mrr"], name="Expansion", marker_color="#5FB8B6"))
    fig_stack.add_trace(go.Bar(x=df["month"], y=-df["churned_mrr"], name="Churn", marker_color="#C94C4C"))
    fig_stack.update_layout(barmode="relative", height=380, xaxis_title="Month",
                              yaxis_title="MRR movement (€)", title="Net New MRR Composition")
    st.plotly_chart(fig_stack, width="stretch")

    st.markdown("#### Rolling 12-month P&L")
    pnl_display = df[[
        "month", "revenue", "cogs", "gross_profit", "rd_cost", "sm_cost", "ga_cost",
        "total_opex", "ebitda", "cash_end",
    ]].copy()
    pnl_display.columns = [
        "Month", "Revenue", "COGS", "Gross Profit", "R&D", "S&M", "G&A",
        "Total Opex", "EBITDA (Burn)", "Cash Balance",
    ]
    for col in pnl_display.columns[1:]:
        pnl_display[col] = pnl_display[col].round(0).map(lambda x: f"€{x:,.0f}")
    st.dataframe(pnl_display, width="stretch", hide_index=True)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download full monthly model (CSV)", csv_buf.getvalue(),
                         file_name=f"{company_name.lower().replace(' ', '_')}_model_{scenario.lower()}.csv",
                         mime="text/csv")


# ----------------------------------------------------------------------
# TAB 3 — UNIT ECONOMICS
# ----------------------------------------------------------------------
with tab_unit:
    st.markdown("#### Unit economics trend")
    col1, col2 = st.columns(2)

    with col1:
        fig_ltv = go.Figure()
        fig_ltv.add_trace(go.Scatter(x=df["month"], y=df["ltv_cac"], mode="lines+markers",
                                       name="LTV:CAC", line=dict(color=TEAL, width=3)))
        fig_ltv.add_hline(y=4, line_dash="dash", line_color=GRAY,
                            annotation_text="4.0x underwriting target")
        fig_ltv.update_layout(title="LTV : CAC Ratio", height=340, xaxis_title="Month", yaxis_title="Ratio (x)")
        st.plotly_chart(fig_ltv, width="stretch")

    with col2:
        fig_payback = go.Figure()
        fig_payback.add_trace(go.Scatter(x=df["month"], y=df["cac_payback_months"], mode="lines+markers",
                                           name="CAC Payback", line=dict(color=DARK, width=3)))
        fig_payback.add_hline(y=18, line_dash="dash", line_color="#C94C4C",
                                annotation_text="18-month efficiency ceiling")
        fig_payback.update_layout(title="CAC Payback Period", height=340,
                                    xaxis_title="Month", yaxis_title="Months")
        st.plotly_chart(fig_payback, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig_gm = go.Figure()
        fig_gm.add_trace(go.Bar(x=df["month"], y=df["gross_margin_pct"] * 100,
                                  marker_color=TEAL, name="Gross Margin %"))
        fig_gm.update_layout(title="Gross Margin % Trend", height=340,
                               xaxis_title="Month", yaxis_title="%")
        st.plotly_chart(fig_gm, width="stretch")

    with col4:
        fig_bm = go.Figure()
        fig_bm.add_trace(go.Bar(x=df["month"], y=df["burn_multiple"],
                                  marker_color="#5FB8B6", name="Burn Multiple"))
        fig_bm.add_hline(y=1, line_dash="dash", line_color=GRAY, annotation_text="1.0x — efficient growth")
        fig_bm.update_layout(title="Burn Multiple (Net Burn ÷ Net New ARR)", height=340,
                               xaxis_title="Month", yaxis_title="x")
        st.plotly_chart(fig_bm, width="stretch")

    st.markdown("#### Customer count evolution")
    df["net_new_customers"] = df["new_customers"] - df["churned_customers"]
    df["cumulative_customers"] = (start_mrr / arpu) + df["net_new_customers"].cumsum()
    fig_cust = go.Figure()
    fig_cust.add_trace(go.Scatter(x=df["month"], y=df["cumulative_customers"], mode="lines+markers",
                                    line=dict(color=TEAL, width=3)))
    fig_cust.update_layout(title="Total Customer Count", height=320, xaxis_title="Month", yaxis_title="Customers")
    st.plotly_chart(fig_cust, width="stretch")


# ----------------------------------------------------------------------
# TAB 4 — CFO COMMENTARY
# ----------------------------------------------------------------------
with tab_ai:
    st.markdown("#### Generate board-pack commentary")
    st.caption(
        "Reads the computed KPIs above and writes a 3-paragraph executive summary: "
        "highlights, risks, and recommendations. Uses Claude if an API key is supplied "
        "in the sidebar; otherwise falls back to a deterministic rules engine."
    )

    if st.button("🤖 Generate CFO Commentary", type="primary"):
        with st.spinner("Analyzing KPIs and drafting commentary..."):
            text, source = generate_ai_commentary(df, meta, api_key_input)
            st.session_state["commentary_text"] = text
            st.session_state["commentary_source"] = source

    if "commentary_text" in st.session_state:
        source_label = "✨ Generated by Claude" if st.session_state["commentary_source"] == "claude" else "⚙️ Generated by rules engine (no API key supplied)"
        st.caption(source_label)
        st.markdown(
            f'<div class="commentary-box">{st.session_state["commentary_text"].replace(chr(10)+chr(10), "<br><br>")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Raw text (for copy/paste into a board deck)")
        st.text_area("Commentary", st.session_state["commentary_text"], height=200, label_visibility="collapsed")

        if FPDF_AVAILABLE:
            kpis_for_pdf = {
                "ARR": fmt_eur(last["arr"]),
                "MRR Growth (MoM)": fmt_pct(last["mrr_growth"]),
                "Net Dollar Retention": fmt_pct(last["ndr"]),
                "Gross Margin": fmt_pct(last["gross_margin_pct"]),
                "CAC Payback": f"{last['cac_payback_months']:.1f} months",
                "LTV : CAC": fmt_x(last["ltv_cac"]),
                "Net Burn": fmt_eur(last["net_burn"]),
                "Cash Runway": fmt_months(last["runway_months"]),
                "Burn Multiple": fmt_x(last["burn_multiple"]),
            }
            pdf_bytes = build_pdf_report(company_name, scenario, kpis_for_pdf,
                                           st.session_state["commentary_text"])
            if pdf_bytes:
                st.download_button("📄 Download Board Report (PDF)", data=pdf_bytes,
                                     file_name=f"{company_name.lower().replace(' ', '_')}_board_report.pdf",
                                     mime="application/pdf")
    else:
        st.info("Click the button above to generate commentary based on the current scenario and assumptions.")


# ----------------------------------------------------------------------
# TAB 5 — FORMULAS & METHODOLOGY
# ----------------------------------------------------------------------
with tab_formulas:
    st.markdown("#### Every formula used in this model")
    st.caption("Full transparency — every number on this dashboard traces back to one of these formulas and the input drivers in the sidebar.")
    formula_df = pd.DataFrame(
        [{"Metric": k, "Formula": v} for k, v in FORMULAS.items()]
    )
    st.dataframe(formula_df, width="stretch", hide_index=True)

    st.markdown("#### Cost allocation methodology")
    st.markdown(
        f"""
        - **COGS** = Revenue × (1 − Gross Margin %)
        - **Headcount cost** splits: **{rd_pct*100:.0f}%** to R&D; of the remainder, 62.5% to S&M and 37.5% to G&A
          (a standard early-stage SaaS blend — adjust the R&D slider in the sidebar to change this split)
        - **CAC spend** (variable, driven by new customers acquired that month) is added to S&M
        - **Other Opex** (tools, facilities, non-headcount G&A) is added to G&A, and grows at the
          monthly opex growth rate to reflect gradual hiring/inflation
        - **New MRR required** each month is solved algebraically so that Beginning MRR + New + Expansion − Churn
          hits the target net growth rate: `New MRR = Beginning MRR × max(Growth Target − Expansion + Churn, 0)`
        """
    )

    st.markdown("#### Current assumption set")
    assumptions_display = pd.DataFrame([{
        "Starting MRR": fmt_eur(start_mrr), "Starting Cash": fmt_eur(start_cash),
        "Headcount": headcount, "Target Net MoM Growth": fmt_pct(growth_target),
        "Gross Churn": fmt_pct(churn), "Expansion": fmt_pct(expansion),
        "ARPU": fmt_eur(arpu), "CAC": fmt_eur(cac), "Gross Margin": fmt_pct(gross_margin),
        "Headcount Cost/mo": fmt_eur(headcount_cost), "Other Opex/mo": fmt_eur(other_opex),
        "Opex Growth/mo": fmt_pct(opex_growth),
    }]).T
    assumptions_display.columns = ["Value"]
    assumptions_display["Value"] = assumptions_display["Value"].astype(str)
    st.dataframe(assumptions_display, width="stretch")

    st.markdown("---")
    st.caption(
        "This is a portfolio project using fictional company data ('Nexus AI'), built to demonstrate "
        "driver-based FP&A modeling, SaaS metrics engineering, and AI-assisted board reporting. "
        "Not real financial data. — Kartikeya Tripathi"
    )
