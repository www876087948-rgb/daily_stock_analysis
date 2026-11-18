# -*- coding: utf-8 -*-
"""Hithink (同花顺) Financial-API fetcher: market breadth primary source.

Market stats (up/down/flat counts, limit-up/down counts, total turnover)
come from the official API: full-market snapshot pages plus the official
limit-up statistics, whose limit-up count overrides the price-rule
computation (official caliber, e.g. 36 vs 37 on 2026-11-17).

Daily K-lines are intentionally NOT served here; the built-in fallback
chain owns them. ``DataFetcherManager.get_market_stats`` tries this
fetcher first explicitly when a key is configured.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd

from .base import BaseFetcher, DataFetchError, calc_market_stats_from_quotes
from .hithink_client import HithinkClient

logger = logging.getLogger(__name__)

_ASIA_SHANGHAI = timezone(timedelta(hours=8))

# Market stats (get_market_stats) paging parameters:
# - snapshot pages of 1000 (endpoint default 100); ~6 pages cover the market;
# - ticker list in a single page (max limit 10000);
# - coverage below 80% is not trustworthy, return None and let the caller
#   fall through to the next source;
# - 3 consecutive page failures abort, so one bad source cannot stall the run.
_SNAPSHOT_PAGE_SIZE = 1000
_TICKERS_PAGE_SIZE = 10000
_MIN_MARKET_COVERAGE = 0.8
_MAX_CONSECUTIVE_PAGE_FAILURES = 3
_MAX_SNAPSHOT_PAGES = 12


def _read_hithink_priority() -> int:
    raw_value = os.getenv("HITHINK_PRIORITY", "6")
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid HITHINK_PRIORITY %r; using 6", raw_value)
        return 6


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


class HithinkFetcher(BaseFetcher):
    """Market breadth via the Hithink official API."""

    name = "HithinkFetcher"
    # Last tier in the generic daily-data chain (this fetcher never serves
    # K-lines); market stats try it first explicitly in DataFetcherManager.
    priority = 6
    allow_empty_daily_data = True

    def __init__(self, api_key: Optional[str] = None, priority: Optional[int] = None) -> None:
        self.priority = priority if priority is not None else _read_hithink_priority()
        self._client = HithinkClient(api_key)

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("HithinkFetcher does not serve daily K-lines")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        return df

    def get_market_stats(self) -> Optional[dict]:
        """Market breadth from Hithink: snapshot pages + official limit-up stats.

        Returns the shared ``calc_market_stats_from_quotes`` dict, with
        ``limit_up_count`` replaced by the official limit-up statistics
        value when available. Returns None on any structural failure so
        the caller falls through to the next source.
        """
        if not self._client.configured:
            return None

        try:
            names = self._all_a_share_names()
        except Exception:
            logger.warning("[Hithink] market stats ticker list failed", exc_info=True)
            return None
        if not names:
            return None

        rows: list[dict[str, Any]] = []
        snapshot_timestamp_ms: Optional[int] = None
        consecutive_failures = 0
        for page in range(_MAX_SNAPSHOT_PAGES):
            offset = page * _SNAPSHOT_PAGE_SIZE
            try:
                page_data = self._client.get(
                    "/api/a-share/prices/snapshot",
                    {"limit": _SNAPSHOT_PAGE_SIZE, "offset": offset},
                )
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "[Hithink] market stats snapshot page %d failed (%d/%d consecutive)",
                    page,
                    consecutive_failures,
                    _MAX_CONSECUTIVE_PAGE_FAILURES,
                    exc_info=True,
                )
                if consecutive_failures >= _MAX_CONSECUTIVE_PAGE_FAILURES:
                    logger.warning("[Hithink] market stats aborted: too many page failures")
                    return None
                continue
            consecutive_failures = 0
            page_data = page_data or {}
            if snapshot_timestamp_ms is None:
                snapshot_timestamp_ms = _safe_int(page_data.get("timestamp"))
            items = page_data.get("item") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                thscode = str(item.get("thscode") or "")
                if not thscode:
                    continue
                rows.append(
                    {
                        "ts_code": thscode,
                        "name": names.get(thscode, ""),
                        "close": item.get("last_price"),
                        "pre_close": item.get("prev_price"),
                        "amount": item.get("turnover"),
                    }
                )
            if len(items) < _SNAPSHOT_PAGE_SIZE:
                break

        if not rows:
            logger.warning("[Hithink] market stats got no quotes")
            return None

        coverage = len(rows) / max(len(names), 1)
        logger.info(
            "[Hithink] market stats coverage %.1f%% (%d/%d symbols)",
            coverage * 100,
            len(rows),
            len(names),
        )
        if coverage < _MIN_MARKET_COVERAGE:
            logger.warning(
                "[Hithink] market stats coverage %.1f%% below %.0f%% threshold; discarding",
                coverage * 100,
                _MIN_MARKET_COVERAGE * 100,
            )
            return None

        df = pd.DataFrame(rows, columns=["ts_code", "name", "close", "pre_close", "amount"])
        stats = calc_market_stats_from_quotes(df)
        if stats is None:
            return None

        # Official limit-up count overrides the price-rule computation.
        # The snapshot timestamp is the trade date of this breadth sample.
        official_limit_up = self._official_limit_up_count(snapshot_timestamp_ms)
        if official_limit_up is not None:
            stats["limit_up_count"] = official_limit_up
            stats["limit_up_official"] = True
        return stats

    def _official_limit_up_count(self, snapshot_timestamp_ms: Optional[int]) -> Optional[int]:
        """Official limit-up count for the snapshot's trade date, if available."""
        trade_date: Optional[str] = None
        if snapshot_timestamp_ms:
            try:
                trade_date = datetime.fromtimestamp(
                    snapshot_timestamp_ms / 1000, tz=_ASIA_SHANGHAI
                ).date().isoformat()
            except (TypeError, ValueError, OverflowError):
                trade_date = None
        if not trade_date:
            trade_date = datetime.now(tz=_ASIA_SHANGHAI).date().isoformat()
        try:
            data = self._client.get(
                "/api/a-share/special-data/limit-up-statistics",
                {"date": trade_date},
            )
            return _safe_int((data or {}).get("limit_up_stock_count"))
        except Exception:
            logger.warning(
                "[Hithink] limit-up statistics failed for %s; using computed counts",
                trade_date,
                exc_info=True,
            )
            return None

    def _all_a_share_names(self) -> dict[str, str]:
        """Full A-share code -> name map from the tickers list endpoint."""
        names: dict[str, str] = {}
        offset = 0
        for _ in range(3):  # ~5500 stocks fit in one 10000 page; 3 is a safety cap
            data = self._client.get(
                "/api/meta/tickers/list",
                {"asset_type": "a-share", "limit": _TICKERS_PAGE_SIZE, "offset": offset},
            )
            items = (data or {}).get("item") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                thscode = str(item.get("thscode") or "")
                name = str(item.get("name") or "")
                if thscode:
                    names[thscode] = name
            if len(items) < _TICKERS_PAGE_SIZE:
                break
            offset += _TICKERS_PAGE_SIZE
        return names
