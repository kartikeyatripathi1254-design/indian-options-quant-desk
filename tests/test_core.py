"""Unit tests for the deterministic parts of the desk (no network, no API calls)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import app

IST = app.IST


def ist(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=IST)


@pytest.mark.parametrize("now,label,gate", [
    (ist(2026, 10, 31, 12, 0), "CLOSED_WEEKEND", "OPEN"),       # Saturday
    (ist(2026, 10, 29, 9, 0), "BEFORE_OPEN", "CLOSED"),
    (ist(2026, 10, 29, 9, 20), "FIRST_15M_CANDLE_OPEN", "CLOSED"),
    (ist(2026, 10, 29, 11, 0), "OPEN", "OPEN"),
    (ist(2026, 10, 29, 16, 0), "CLOSED_AFTER_MARKET", "OPEN"),
])
def test_session_and_gate(now, label, gate):
    assert app.session_label(now) == label
    assert app.ideas_gate(now) == gate


def _closes(n, start="2026-08-03", drift=0.001, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    prices = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    return pd.DataFrame({"Close": prices}, index=idx)


def test_hv20_matches_manual_calculation():
    df = _closes(40)
    now = datetime.combine(df.index[-1].date() + pd.Timedelta(days=1), datetime.min.time()).replace(hour=18, tzinfo=IST)
    m = app.compute_metrics(df, now)
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna().tail(20)
    assert m["hv_20d_annualised_pct"] == pytest.approx(log_ret.std(ddof=1) * np.sqrt(252) * 100, abs=0.01)
    assert m["data_status"] == "OK"
    assert m["today_partial_bar_dropped"] is False


def test_partial_bar_dropped_during_session():
    df = _closes(40)
    last_day = df.index[-1].date()
    now = datetime(last_day.year, last_day.month, last_day.day, 11, 0, tzinfo=IST)
    m = app.compute_metrics(df, now)
    assert m["today_partial_bar_dropped"] is True
    assert m["last_bar_date"] == str(df.index[-2].date())


def test_stale_data_flagged():
    df = _closes(40)
    now = datetime.combine(df.index[-1].date() + pd.Timedelta(days=10), datetime.min.time()).replace(hour=18, tzinfo=IST)
    assert app.compute_metrics(df, now)["data_status"] == "STALE"


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        app.compute_metrics(_closes(15), ist(2026, 12, 1, 18, 0))


def _report(verdict="TAKE_TRADE", invalidation=24000.0, name="Nifty 50"):
    return app.OptionsMasterReport(
        macro_market_summary="test",
        unknowns=[],
        trade_recommendations=[app.Recommendation(
            instrument_name=name, asset_type="Index", recommended_strategy="Bear put spread",
            market_outlook="x", entry_condition="x", invalidation_level=invalidation,
            conditional_idea="x", risk_assessment_details="x", confidence="MEDIUM", verdict=verdict,
        )],
    )


MD_OK = {"Nifty 50": {"nse_lot_size": 25, "data_status": "OK"}}


def test_trade_kept_when_all_rules_pass():
    rows = app.enforce_rules(_report(), MD_OK, gate="OPEN", chain_provided=True)
    assert rows[0]["final_verdict"] == "TAKE_TRADE"
    assert rows[0]["rule_overrides"] == ""


@pytest.mark.parametrize("kwargs,reason", [
    ({"gate": "CLOSED", "chain_provided": True}, "09:30"),
    ({"gate": "OPEN", "chain_provided": False}, "option-chain"),
])
def test_trade_downgraded_by_rules(kwargs, reason):
    rows = app.enforce_rules(_report(), MD_OK, **kwargs)
    assert rows[0]["final_verdict"] == "STAY_FLAT"
    assert reason in rows[0]["rule_overrides"]


def test_trade_downgraded_without_invalidation_or_fresh_data():
    rows = app.enforce_rules(_report(invalidation=None), MD_OK, gate="OPEN", chain_provided=True)
    assert rows[0]["final_verdict"] == "STAY_FLAT"
    stale = {"Nifty 50": {"nse_lot_size": 25, "data_status": "STALE"}}
    rows = app.enforce_rules(_report(), stale, gate="OPEN", chain_provided=True)
    assert "STALE" in rows[0]["rule_overrides"]


def test_unknown_instrument_is_flagged():
    rows = app.enforce_rules(_report(name="Made Up Index"), MD_OK, gate="OPEN", chain_provided=True)
    assert rows[0]["final_verdict"] == "STAY_FLAT"
    assert "not recognised" in rows[0]["rule_overrides"]


def test_spread_risk():
    credit = app.spread_risk("Credit spread (you receive premium)", width_pts=100, premium_pts=30, lot=25)
    assert credit == {"max_profit_per_lot": 750.0, "max_loss_per_lot": 1750.0}
    debit = app.spread_risk("Debit spread (you pay premium)", width_pts=100, premium_pts=30, lot=25)
    assert debit == {"max_profit_per_lot": 1750.0, "max_loss_per_lot": 750.0}


def test_nse_holiday_is_closed():
    assert app.session_label(ist(2026, 11, 10, 11, 0)) == "CLOSED_HOLIDAY"   # Diwali Balipratipada
    assert app.session_label(ist(2026, 11, 11, 11, 0)) == "OPEN"
    assert all(d.weekday() < 5 for d in app.NSE_HOLIDAYS)


def test_call_limit_counts_only_recorded_calls():
    counter = {}
    assert not app.limit_reached(2, counter)
    app.record_call(counter)
    assert not app.limit_reached(2, counter)
    app.record_call(counter)
    assert app.limit_reached(2, counter)


class _FakeMessages:
    def __init__(self, stop_reason, parsed):
        self.stop_reason, self.parsed, self.kwargs = stop_reason, parsed, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return type("Msg", (), {"stop_reason": self.stop_reason, "parsed_output": self.parsed})()


def _fake_client(monkeypatch, stop_reason, parsed):
    messages = _FakeMessages(stop_reason, parsed)
    monkeypatch.setattr(app.anthropic, "Anthropic",
                        lambda **_: type("Client", (), {"messages": messages})())
    return messages


def test_call_claude_uses_structured_output_without_forced_tool(monkeypatch):
    messages = _fake_client(monkeypatch, "end_turn", _report())
    assert app.call_claude("key", {}).trade_recommendations[0].instrument_name == "Nifty 50"
    assert messages.kwargs["output_format"] is app.OptionsMasterReport
    assert "tool_choice" not in messages.kwargs


@pytest.mark.parametrize("stop_reason,parsed", [("refusal", None), ("max_tokens", None), ("end_turn", None)])
def test_call_claude_raises_without_report(monkeypatch, stop_reason, parsed):
    _fake_client(monkeypatch, stop_reason, parsed)
    with pytest.raises(ValueError):
        app.call_claude("key", {})
