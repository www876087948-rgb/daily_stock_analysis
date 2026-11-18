# -*- coding: utf-8 -*-
"""Tests for the Hithink (同花顺) fundamental bundle adapter (offline, mocked)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from data_provider.base import _hithink_bundle_has_content
from data_provider.hithink_client import HithinkApiError
from data_provider.hithink_fundamental_adapter import (
    HithinkFundamentalAdapter,
    _flatten_indicators,
    _report_param,
)

_ASIA_SHANGHAI = timezone(timedelta(hours=8))


def test_report_param_mapping() -> None:
    assert _report_param(2026, "Q1") == "2026-1"
    assert _report_param(2026, "Q2") == "2026-2"
    assert _report_param(2026, "Q3") == "2026-3"
    assert _report_param(2025, "FY") == "2025-4"
    assert _report_param(2025, "Q4") is None
    assert _report_param(None, "Q3") is None


def test_flatten_indicators() -> None:
    data = {
        "abilities": [
            {
                "ability": "growth",
                "indicators": [
                    {"index_id": "operating_income_yoy_growth_ratio", "value": "-1.21"},
                    {"index_id": "net_profit_yoy_growth_ratio", "value": "-4.53"},
                ],
            },
            {
                "ability": "profitability",
                "indicators": [{"index_id": "sale_gross_margin", "value": "91.20"}],
            },
        ]
    }
    values = _flatten_indicators(data)
    assert values["operating_income_yoy_growth_ratio"] == "-1.21"
    assert values["sale_gross_margin"] == "91.20"
    assert _flatten_indicators({}) == {}
    assert _flatten_indicators(None) == {}


def _fake_adapter_get_factory(captured: dict):
    """Fake HithinkClient.get returning canned financial payloads."""
    now = datetime.now(tz=_ASIA_SHANGHAI).date()
    recent_ex_ms = int(
        datetime(now.year, now.month, 1, tzinfo=_ASIA_SHANGHAI).timestamp() * 1000
    )
    old_ex_ms = int(datetime(2024, 1, 10, tzinfo=_ASIA_SHANGHAI).timestamp() * 1000)

    def fake_get(path: str, params: dict | None = None):
        captured[path] = dict(params or {})
        if path == "/api/a-share/financials/income-statements":
            return {
                "item": [
                    {
                        "thscode": "600519.SH",
                        "fiscal_year": 2025,
                        "fiscal_period": "FY",
                        "report_date_ms": 1776355200000,
                        "operating_income": 168838102514.79,
                        "parent_holder_net_profit": 82320067101.68,
                        "basic_eps": 65.74,
                    }
                ]
            }
        if path == "/api/a-share/financials/indicators":
            return {
                "abilities": [
                    {
                        "ability": "growth",
                        "indicators": [
                            {"index_id": "operating_income_yoy_growth_ratio", "value": "-1.21"},
                            {"index_id": "net_profit_yoy_growth_ratio", "value": "-4.53"},
                        ],
                    },
                    {
                        "ability": "profitability",
                        "indicators": [
                            {"index_id": "sale_gross_margin", "value": "91.20"},
                            {"index_id": "index_weighted_avg_roe", "value": "32.53"},
                        ],
                    },
                ]
            }
        if path == "/api/a-share/financials/cash-flow-statements":
            return {"item": [{"act_cash_flow_net": 61522204989.35}]}
        if path == "/api/a-share/corporate-actions/adjustment-factors":
            return {
                "item": [
                    {"ex_date_ms": recent_ex_ms, "dividend_per_share": 27.993},
                    {"ex_date_ms": old_ex_ms, "dividend_per_share": 1.0},
                ]
            }
        raise AssertionError(f"unexpected path {path}")

    return fake_get


def test_bundle_happy_path_maps_all_blocks() -> None:
    adapter = HithinkFundamentalAdapter(api_key="test-key")
    captured: dict = {}
    adapter._client.get = _fake_adapter_get_factory(captured)
    bundle = adapter.get_fundamental_bundle("600519")

    assert bundle["status"] == "partial"
    assert bundle["growth"] == {
        "revenue_yoy": -1.21,
        "net_profit_yoy": -4.53,
        "roe": 32.53,
        "gross_margin": 91.2,
    }
    report = bundle["earnings"]["financial_report"]
    assert report["revenue"] == pytest.approx(168838102514.79)
    assert report["net_profit_parent"] == pytest.approx(82320067101.68)
    assert report["basic_eps"] == pytest.approx(65.74)
    assert report["operating_cash_flow"] == pytest.approx(61522204989.35)
    assert report["roe"] == pytest.approx(32.53)
    assert report["report_date"] == "2026-04-17"
    dividend = bundle["earnings"]["dividend"]
    assert dividend["ttm_cash_dividend_per_share"] == pytest.approx(27.993)
    assert dividend["ttm_event_count"] == 1
    assert dividend["coverage"] == "cash_dividend_pre_tax"
    assert "financial_report:hithink" in bundle["source_chain"]
    assert "growth:hithink" in bundle["source_chain"]
    assert "cash_flow:hithink" in bundle["source_chain"]
    assert "dividend:hithink" in bundle["source_chain"]
    # FY 2025 -> indicators report "2025-4".
    assert captured["/api/a-share/financials/indicators"]["report"] == "2025-4"
    assert captured["/api/a-share/financials/indicators"]["thscode"] == "600519.SH"


def test_bundle_without_key_is_not_supported() -> None:
    adapter = HithinkFundamentalAdapter(api_key="")
    bundle = adapter.get_fundamental_bundle("600519")
    assert bundle["status"] == "not_supported"
    assert bundle["errors"]


def test_bundle_partial_when_indicators_fail() -> None:
    adapter = HithinkFundamentalAdapter(api_key="test-key")
    captured: dict = {}
    real_fake = _fake_adapter_get_factory(captured)

    def failing_indicators(path: str, params: dict | None = None):
        if path == "/api/a-share/financials/indicators":
            raise HithinkApiError("indicators down")
        return real_fake(path, params)

    adapter._client.get = failing_indicators
    bundle = adapter.get_fundamental_bundle("600519")
    assert bundle["status"] == "partial"
    assert bundle["growth"] == {}
    assert bundle["earnings"]["financial_report"]["revenue"] == pytest.approx(168838102514.79)
    assert any("indicators failed" in err for err in bundle["errors"])


def test_bundle_not_supported_when_all_calls_fail() -> None:
    adapter = HithinkFundamentalAdapter(api_key="test-key")
    adapter._client.get = lambda path, params=None: (_ for _ in ()).throw(
        HithinkApiError("all down")
    )
    bundle = adapter.get_fundamental_bundle("600519")
    assert bundle["status"] == "not_supported"
    # Income failure means no report period, so the indicators step is
    # skipped (no error appended for it): 3 errors, not 4.
    assert len(bundle["errors"]) == 3
    assert any("income-statements failed" in err for err in bundle["errors"])
    assert any("cash-flow-statements failed" in err for err in bundle["errors"])
    assert any("dividend failed" in err for err in bundle["errors"])


def test_hithink_bundle_has_content_gate() -> None:
    assert _hithink_bundle_has_content({}) is False
    assert _hithink_bundle_has_content(None) is False
    assert _hithink_bundle_has_content({"growth": {"revenue_yoy": -1.21}}) is True
    assert _hithink_bundle_has_content({"growth": {"revenue_yoy": None}}) is False
    assert (
        _hithink_bundle_has_content(
            {"earnings": {"financial_report": {"report_date": "2026-04-16", "revenue": 100.0}}}
        )
        is True
    )
    assert (
        _hithink_bundle_has_content(
            {"earnings": {"financial_report": {"report_date": "2026-04-16"}}}
        )
        is False
    )
    assert _hithink_bundle_has_content({"earnings": {"dividend": {}}}) is False
