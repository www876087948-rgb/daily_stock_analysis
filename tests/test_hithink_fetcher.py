# -*- coding: utf-8 -*-
"""Tests for the Hithink (同花顺) market breadth fetcher (offline, mocked)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_provider.base import DataFetchError
from data_provider.hithink_client import (
    HithinkApiError,
    HithinkRateLimitError,
    from_thscode,
    to_thscode,
)
from data_provider.hithink_fetcher import HithinkFetcher


def _read_priority_from_fresh_process(value: str | None) -> int:
    env = os.environ.copy()
    if value is None:
        env.pop("HITHINK_PRIORITY", None)
    else:
        env["HITHINK_PRIORITY"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from data_provider.hithink_fetcher import HithinkFetcher; "
                "print(HithinkFetcher().priority)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(result.stdout.strip().splitlines()[-1])


def test_hithink_priority_defaults_to_last_tier_and_allows_override() -> None:
    assert _read_priority_from_fresh_process(None) == 6
    assert _read_priority_from_fresh_process("2") == 2


def test_to_thscode_mapping() -> None:
    assert to_thscode("600519") == "600519.SH"
    assert to_thscode("688001") == "688001.SH"
    assert to_thscode("000001") == "000001.SZ"
    assert to_thscode("300033") == "300033.SZ"
    assert to_thscode("430047") == "430047.BJ"
    assert to_thscode("123") is None
    assert to_thscode("600519.SH") == "600519.SH"
    assert from_thscode("600519.SH") == "600519"


def _mock_response(status_code: int, payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_client_envelope_errors() -> None:
    from data_provider.hithink_client import HithinkClient

    client = HithinkClient(api_key="test-key")
    with patch(
        "data_provider.hithink_client.requests.get",
        return_value=_mock_response(200, {"code": 1001, "message": "bad param"}),
    ):
        with pytest.raises(HithinkApiError):
            client.get("/api/meta/tickers/search", {"q": "600519"})
    with patch(
        "data_provider.hithink_client.requests.get",
        return_value=_mock_response(429, {"code": 4001, "message": "limited"}),
    ):
        with pytest.raises(HithinkRateLimitError):
            client.get("/api/a-share/prices/snapshot")
    with patch(
        "data_provider.hithink_client.requests.get",
        return_value=_mock_response(200, {"code": 4001, "message": "limited"}),
    ):
        with pytest.raises(HithinkRateLimitError):
            client.get("/api/a-share/prices/snapshot")
    with pytest.raises(HithinkApiError):
        HithinkClient(api_key="").get("/api/meta/tickers/search")


def _fake_hithink_get_factory(*, names, snapshot_items, limit_up_count, fail_snapshot=False):
    """Fake HithinkClient.get dispatching on path for market-stats tests."""
    calls: list[str] = []

    def fake_get(path: str, params: dict | None = None):
        calls.append(path)
        if path == "/api/meta/tickers/list":
            return {"item": [{"thscode": code, "name": name} for code, name in names.items()]}
        if path == "/api/a-share/prices/snapshot":
            if fail_snapshot:
                raise HithinkApiError("boom")
            offset = (params or {}).get("offset", 0)
            if offset > 0:
                return {"timestamp": 1794977705000, "item": []}
            return {"timestamp": 1794977705000, "item": snapshot_items}
        if path == "/api/a-share/special-data/limit-up-statistics":
            return {"limit_up_stock_count": limit_up_count}
        raise AssertionError(f"unexpected path {path}")

    fake_get.calls = calls  # type: ignore[attr-defined]
    return fake_get


def test_market_stats_happy_path_uses_official_limit_up() -> None:
    fetcher = HithinkFetcher(api_key="test-key")
    names = {
        "600519.SH": "贵州茅台",
        "000001.SZ": "平安银行",
        "600001.SH": "ST测试",
    }
    snapshot_items = [
        {"thscode": "600519.SH", "last_price": 1450.72, "prev_price": 1457.7, "turnover": 1832138900},
        {"thscode": "000001.SZ", "last_price": 12.2, "prev_price": 12.25, "turnover": 946416070},
        # ST stock at its 5% limit-up price: computed limit-up = 1, official = 2.
        {"thscode": "600001.SH", "last_price": 10.5, "prev_price": 10.0, "turnover": 500000000},
    ]
    fetcher._client.get = _fake_hithink_get_factory(
        names=names, snapshot_items=snapshot_items, limit_up_count=2
    )
    stats = fetcher.get_market_stats()
    assert stats is not None
    assert stats["up_count"] == 1
    assert stats["down_count"] == 2
    assert stats["flat_count"] == 0
    assert stats["limit_up_count"] == 2
    assert stats["limit_up_official"] is True
    assert stats["limit_down_count"] == 0
    assert stats["total_amount"] == pytest.approx(32.7855497, rel=1e-6)


def test_market_stats_low_coverage_returns_none() -> None:
    fetcher = HithinkFetcher(api_key="test-key")
    names = {f"60000{i}.SH": f"股票{i}" for i in range(10)}
    snapshot_items = [
        {"thscode": f"60000{i}.SH", "last_price": 10.0, "prev_price": 9.0, "turnover": 1000000}
        for i in range(5)
    ]
    fetcher._client.get = _fake_hithink_get_factory(
        names=names, snapshot_items=snapshot_items, limit_up_count=1
    )
    assert fetcher.get_market_stats() is None


def test_market_stats_snapshot_failures_return_none() -> None:
    fetcher = HithinkFetcher(api_key="test-key")
    names = {"600519.SH": "贵州茅台"}
    fetcher._client.get = _fake_hithink_get_factory(
        names=names, snapshot_items=[], limit_up_count=1, fail_snapshot=True
    )
    assert fetcher.get_market_stats() is None


def test_market_stats_without_key_returns_none() -> None:
    fetcher = HithinkFetcher(api_key="")
    assert fetcher.get_market_stats() is None


def test_fetch_raw_data_raises_fast() -> None:
    fetcher = HithinkFetcher(api_key="test-key")
    with pytest.raises(DataFetchError):
        fetcher._fetch_raw_data("600519", "2026-11-01", "2026-11-18")
