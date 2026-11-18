# -*- coding: utf-8 -*-
"""Hithink (同花顺) fundamental bundle adapter.

Same return contract as
:class:`data_provider.fundamental_adapter.AkshareFundamentalAdapter`
``get_fundamental_bundle``: ``growth`` / ``earnings.financial_report`` /
``earnings.dividend`` blocks, partial-tolerant (each call may fail
independently). The manager tries this adapter before AkShare when a
Hithink key is configured, and falls back to AkShare when the bundle has
no usable growth / financial-report content.

Data mapping (all amounts in CNY, percentages as plain numbers, e.g.
``89.12`` means 89.12%):
- revenue / net profit / EPS: latest quarterly income statement
  (``operating_income`` / ``parent_holder_net_profit`` / ``basic_eps``);
- YoY / ROE / gross margin: financial indicators for the same report
  period (``operating_income_yoy_growth_ratio`` /
  ``net_profit_yoy_growth_ratio`` / ``index_weighted_avg_roe`` /
  ``sale_gross_margin``);
- operating cash flow: latest quarterly cash-flow statement
  (``act_cash_flow_net``);
- TTM cash dividend: corporate-actions adjustment factors in the last
  365 days, same payload shape as the AkShare dividend builder.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .hithink_client import HithinkClient, to_thscode

logger = logging.getLogger(__name__)

_ASIA_SHANGHAI = timezone(timedelta(hours=8))
_FISCAL_PERIOD_TO_REPORT = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
_INDICATOR_IDS = {
    "revenue_yoy": "operating_income_yoy_growth_ratio",
    "net_profit_yoy": "net_profit_yoy_growth_ratio",
    "gross_margin": "sale_gross_margin",
    "roe": "index_weighted_avg_roe",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _ms_to_date(value: Any) -> Optional[str]:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(float(value) / 1000, tz=_ASIA_SHANGHAI).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _report_param(fiscal_year: Any, fiscal_period: Any) -> Optional[str]:
    """``(2026, 'Q3')`` -> ``'2026-3'`` (indicators endpoint format)."""
    try:
        year = int(fiscal_year)
    except (TypeError, ValueError):
        return None
    quarter = _FISCAL_PERIOD_TO_REPORT.get(str(fiscal_period or "").strip().upper())
    if quarter is None:
        return None
    return f"{year}-{quarter}"


def _flatten_indicators(data: Any) -> Dict[str, Any]:
    """``abilities[].indicators[]`` -> ``{index_id: value}``."""
    values: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return values
    for ability in data.get("abilities") or []:
        if not isinstance(ability, dict):
            continue
        for indicator in ability.get("indicators") or []:
            if not isinstance(indicator, dict):
                continue
            index_id = indicator.get("index_id")
            if index_id:
                values[str(index_id)] = indicator.get("value")
    return values


class HithinkFundamentalAdapter:
    """Fundamental bundle from the Hithink official API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._client = HithinkClient(api_key)

    @property
    def configured(self) -> bool:
        return self._client.configured

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """Return normalized fundamental blocks from Hithink."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }
        if not self._client.configured:
            result["errors"].append("hithink api key not configured")
            return result
        thscode = to_thscode(stock_code)
        if not thscode:
            result["errors"].append(f"unsupported stock code: {stock_code}")
            return result

        # 1. Latest quarterly income statement: revenue / net profit / EPS.
        report: Optional[str] = None
        try:
            income = self._client.get(
                "/api/a-share/financials/income-statements",
                {"thscode": thscode, "period": "quarterly", "limit": 1},
            )
            items = (income or {}).get("item") or []
            if items:
                row = items[0] if isinstance(items[0], dict) else {}
                financial_report = {
                    "report_date": _ms_to_date(row.get("report_date_ms")),
                    "revenue": _safe_float(row.get("operating_income")),
                    "net_profit_parent": _safe_float(row.get("parent_holder_net_profit")),
                    "basic_eps": _safe_float(row.get("basic_eps")),
                }
                if any(v is not None for k, v in financial_report.items() if k != "report_date"):
                    result["earnings"]["financial_report"] = financial_report
                    result["source_chain"].append("financial_report:hithink")
                report = _report_param(row.get("fiscal_year"), row.get("fiscal_period"))
        except Exception as exc:
            result["errors"].append(f"income-statements failed: {exc}")

        # 2. Indicators for the same report period: YoY / ROE / margin.
        if report:
            try:
                indicators = self._client.get(
                    "/api/a-share/financials/indicators",
                    {"thscode": thscode, "report": report},
                )
                values = _flatten_indicators(indicators)
                growth = {
                    "revenue_yoy": _safe_float(values.get(_INDICATOR_IDS["revenue_yoy"])),
                    "net_profit_yoy": _safe_float(values.get(_INDICATOR_IDS["net_profit_yoy"])),
                    "roe": _safe_float(values.get(_INDICATOR_IDS["roe"])),
                    "gross_margin": _safe_float(values.get(_INDICATOR_IDS["gross_margin"])),
                }
                if any(v is not None for v in growth.values()):
                    result["growth"] = growth
                    result["source_chain"].append("growth:hithink")
                    financial_report = result["earnings"].get("financial_report")
                    if isinstance(financial_report, dict) and growth.get("roe") is not None:
                        financial_report["roe"] = growth["roe"]
            except Exception as exc:
                result["errors"].append(f"indicators failed: {exc}")

        # 3. Latest quarterly cash-flow statement: operating cash flow.
        try:
            cashflow = self._client.get(
                "/api/a-share/financials/cash-flow-statements",
                {"thscode": thscode, "period": "quarterly", "limit": 1},
            )
            items = (cashflow or {}).get("item") or []
            if items and isinstance(items[0], dict):
                ocf = _safe_float(items[0].get("act_cash_flow_net"))
                financial_report = result["earnings"].get("financial_report")
                if ocf is not None and isinstance(financial_report, dict):
                    financial_report["operating_cash_flow"] = ocf
                    result["source_chain"].append("cash_flow:hithink")
        except Exception as exc:
            result["errors"].append(f"cash-flow-statements failed: {exc}")

        # 4. TTM cash dividend from adjustment factors (last 365 days).
        try:
            dividend = self._ttm_dividend(thscode)
            if dividend:
                result["earnings"]["dividend"] = dividend
                result["source_chain"].append("dividend:hithink")
        except Exception as exc:
            result["errors"].append(f"dividend failed: {exc}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def _ttm_dividend(self, thscode: str) -> Dict[str, Any]:
        """TTM cash dividend payload, same shape as the AkShare builder."""
        now_date = datetime.now(tz=_ASIA_SHANGHAI).date()
        ttm_start = now_date - timedelta(days=365)
        window_start = now_date - timedelta(days=400)
        data = self._client.get(
            "/api/a-share/corporate-actions/adjustment-factors",
            {
                "thscode": thscode,
                "from": window_start.isoformat(),
                "to": now_date.isoformat(),
            },
        )
        items = (data or {}).get("item") or []
        events: List[Dict[str, Any]] = []
        seen: set = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            ex_date = _ms_to_date(item.get("ex_date_ms"))
            if not ex_date:
                continue
            per_share = _safe_float(item.get("dividend_per_share"))
            if per_share is None or per_share <= 0:
                continue
            if ex_date > now_date.isoformat():
                continue
            key = (ex_date, round(per_share, 6))
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "event_date": ex_date,
                    "ex_dividend_date": ex_date,
                    "record_date": None,
                    "announcement_date": None,
                    "cash_dividend_per_share": round(per_share, 6),
                    "is_pre_tax": True,
                }
            )
        if not events:
            return {}
        events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
        ttm_events = [e for e in events if ttm_start.isoformat() <= (e.get("event_date") or "") <= now_date.isoformat()]
        return {
            "events": events[:5],
            "ttm_event_count": len(ttm_events),
            "ttm_cash_dividend_per_share": (
                round(sum(float(e.get("cash_dividend_per_share") or 0.0) for e in ttm_events), 6)
                if ttm_events
                else None
            ),
            "coverage": "cash_dividend_pre_tax",
            "as_of": now_date.isoformat(),
        }
