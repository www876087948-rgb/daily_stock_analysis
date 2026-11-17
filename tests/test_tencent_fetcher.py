# -*- coding: utf-8 -*-
"""Tests for Tencent direct daily K-line fetcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests import Response

from data_provider.base import calc_market_stats_from_quotes
from data_provider.tencent_fetcher import (
    TencentFetcher,
    _is_star_market,
    _lots_to_shares,
    _parse_quote_fields,
    _to_tencent_symbol,
)


def _read_priority_from_fresh_process(value: str | None) -> int:
    env = os.environ.copy()
    if value is None:
        env.pop("TENCENT_PRIORITY", None)
    else:
        env["TENCENT_PRIORITY"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from data_provider.tencent_fetcher import TencentFetcher; "
                "print(TencentFetcher().priority)"
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


def test_tencent_priority_defaults_to_last_resort_and_allows_override() -> None:
    assert _read_priority_from_fresh_process(None) == 5
    assert _read_priority_from_fresh_process("2") == 2


def test_tencent_priority_honors_env_set_after_package_import() -> None:
    env = os.environ.copy()
    env.pop("TENCENT_PRIORITY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import data_provider; "
                "import os; "
                "os.environ['TENCENT_PRIORITY'] = '0'; "
                "from data_provider import DataFetcherManager; "
                "manager = DataFetcherManager(); "
                "print([(f.name, f.priority) for f in manager._get_fetchers_snapshot() if f.name == 'TencentFetcher'][0][1])"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert int(result.stdout.strip().splitlines()[-1]) == 0


def test_tencent_priority_honors_dotenv_when_manager_loads_config_after_package_import(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TENCENT_PRIORITY=0\n", encoding="utf-8")

    env = os.environ.copy()
    env.pop("TENCENT_PRIORITY", None)
    env["ENV_FILE"] = str(env_file)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import data_provider; "
                "from data_provider import DataFetcherManager; "
                "manager = DataFetcherManager(); "
                "print([(f.name, f.priority) for f in manager._get_fetchers_snapshot() if f.name == 'TencentFetcher'][0][1])"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert int(result.stdout.strip().splitlines()[-1]) == 0


def test_tencent_symbol_conversion_supports_a_share_markets() -> None:
    assert _to_tencent_symbol("600519") == "sh600519"
    assert _to_tencent_symbol("000001") == "sz000001"
    assert _to_tencent_symbol("920748") == "bj920748"
    assert _to_tencent_symbol("sh000016") == "sh000016"
    assert _to_tencent_symbol("000016.SH") == "sh000016"
    assert _to_tencent_symbol("sz399001") == "sz399001"


def test_tencent_fetcher_preserves_explicit_index_market_for_daily_request() -> None:
    payload = {
        "data": {
            "sh000016": {
                "qfqday": [
                    ["2026-08-21", "100", "101", "102", "99", "1000", "101000"]
                ]
            }
        }
    }
    response = MagicMock()
    response.json.return_value = payload

    with patch("data_provider.tencent_fetcher.requests.get", return_value=response) as request:
        df = TencentFetcher().get_daily_data(
            "sh000016", start_date="2026-08-01", end_date="2026-08-21"
        )

    assert not df.empty
    assert request.call_args.kwargs["params"]["param"].startswith("sh000016,day,")


def test_tencent_fetcher_get_stock_name_uses_lightweight_quote_request() -> None:
    response = MagicMock()
    response.text = 'v_sh000016="1~上证50~000016~0";'

    with patch("data_provider.tencent_fetcher.requests.get", return_value=response) as request:
        name = TencentFetcher().get_stock_name("sh000016")

    assert name == "上证50"
    assert request.call_args.args[0] == "https://qt.gtimg.cn/q=sh000016"
    response.raise_for_status.assert_called_once_with()


def test_tencent_fetcher_get_stock_name_rejects_mismatched_response_code() -> None:
    response = MagicMock()
    response.text = 'v_sh000016="1~沪深300~000300~0";'

    with patch("data_provider.tencent_fetcher.requests.get", return_value=response):
        name = TencentFetcher().get_stock_name("sh000016")

    assert name is None


def test_tencent_fetcher_get_stock_name_decodes_real_gbk_response() -> None:
    response = Response()
    response.status_code = 200
    response.url = "https://qt.gtimg.cn/q=sh000016"
    response.encoding = "ISO-8859-1"
    response._content = 'v_sh000016="1~上证50~000016~0";'.encode("gbk")

    with patch("data_provider.tencent_fetcher.requests.get", return_value=response):
        name = TencentFetcher().get_stock_name("sh000016")

    assert name == "上证50"


def test_tencent_fetcher_parses_qfq_daily_response() -> None:
    payload = {
        "data": {
            "sz000001": {
                "qfqday": [
                    ["2026-05-06", "10.00", "10.50", "10.80", "9.90", "12345", "67890"],
                    ["2026-05-07", "10.50", "10.70", "10.90", "10.30", "22345", "77890"],
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    fetcher = TencentFetcher()
    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = fetcher.get_daily_data("000001", start_date="2026-05-01", end_date="2026-05-10")

    assert captured["url"] == "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    assert captured["params"]["param"].startswith("sz000001,day,2026-05-01,2026-05-10,")
    assert captured["params"]["param"].endswith(",qfq")
    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        "ma5",
        "ma10",
        "ma20",
        "volume_ratio",
    ]
    assert len(df) == 2
    assert float(df.iloc[0]["close"]) == 10.5
    assert float(df.iloc[0]["volume"]) == 1234500.0
    assert float(df.iloc[1]["amount"]) == 77890.0


def test_tencent_fetcher_requests_explicit_historical_date_window() -> None:
    payload = {
        "data": {
            "sz000001": {
                "qfqday": [
                    ["2020-05-04", "8.00", "8.20", "8.40", "7.80", "5000", "20000"],
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    fetcher = TencentFetcher()
    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = fetcher.get_daily_data("000001", start_date="2020-05-01", end_date="2020-05-31")

    assert captured["url"] == "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    assert ",day,2020-05-01,2020-05-31," in captured["params"]["param"]
    assert captured["params"]["param"].endswith(",qfq")
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 8.2
    assert float(df.iloc[0]["volume"]) == 500000.0


def test_tencent_fetcher_preserves_amount_column_when_missing() -> None:
    payload = {
        "data": {
            "sh600519": {
                "qfqday": [
                    ["2026-05-06", "100.00", "101.00", "102.00", "99.00", "1000"],
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    with patch("data_provider.tencent_fetcher.requests.get", return_value=FakeResponse()):
        df = TencentFetcher().get_daily_data("600519", start_date="2026-05-01", end_date="2026-05-10")

    assert "amount" in df.columns
    assert pd.isna(df.iloc[0]["amount"])
    assert float(df.iloc[0]["volume"]) == 100000.0


def test_tencent_fetcher_returns_empty_frame_for_empty_history() -> None:
    payload = {"data": {"sz000001": {"qfqday": []}}}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    with patch("data_provider.tencent_fetcher.requests.get", return_value=FakeResponse()):
        df = TencentFetcher().get_daily_data("000001", start_date="2026-05-01", end_date="2026-05-10")

    assert df.empty


def test_tencent_fetcher_keeps_short_history_when_cap_not_hit() -> None:
    payload = {
        "data": {
            "sz000001": {
                "qfqday": [
                    ["2023-01-03", "10.00", "10.50", "10.80", "9.90", "12345", "67890"],
                    ["2023-01-04", "10.50", "10.70", "10.90", "10.30", "22345", "77890"],
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = TencentFetcher().get_daily_data("000001", start_date="2020-01-01", end_date="2026-05-10")

    assert ",day,2020-01-01,2026-05-10,800,qfq" in captured["params"]["param"]
    assert len(df) == 2
    assert float(df.iloc[0]["close"]) == 10.5


def test_tencent_fetcher_keeps_near_cap_short_history_for_new_listing() -> None:
    rows = [
        [
            day.strftime("%Y-%m-%d"),
            "10.00",
            "10.50",
            "10.80",
            "9.90",
            str(10000 + index),
            str(20000 + index),
        ]
        for index, day in enumerate(pd.date_range("2024-01-03", periods=799, freq="D"))
    ]
    payload = {"data": {"sz000001": {"qfqday": rows}}}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = TencentFetcher().get_daily_data("000001", start_date="2020-01-01", end_date="2026-05-10")

    assert ",day,2020-01-01,2026-05-10,800,qfq" in captured["params"]["param"]
    assert len(df) == 799
    assert float(df.iloc[0]["close"]) == 10.5


def test_tencent_fetcher_keeps_capped_history_when_start_is_weekend() -> None:
    rows = [
        [
            day.strftime("%Y-%m-%d"),
            "10.00",
            "10.50",
            "10.80",
            "9.90",
            str(10000 + index),
            str(20000 + index),
        ]
        for index, day in enumerate(pd.bdate_range("2024-03-04", periods=800))
    ]
    payload = {"data": {"sz000001": {"qfqday": rows}}}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = TencentFetcher().get_daily_data("000001", start_date="2024-03-02", end_date="2027-05-10")

    assert ",day,2024-03-02,2027-05-10,800,qfq" in captured["params"]["param"]
    assert len(df) == 800
    assert pd.Timestamp(df.iloc[0]["date"]).strftime("%Y-%m-%d") == "2024-03-04"


def test_tencent_fetcher_rejects_capped_incomplete_history() -> None:
    rows = [
        [
            day.strftime("%Y-%m-%d"),
            "10.00",
            "10.50",
            "10.80",
            "9.90",
            str(10000 + index),
            str(20000 + index),
        ]
        for index, day in enumerate(pd.date_range("2023-01-03", periods=800, freq="D"))
    ]
    payload = {"data": {"sz000001": {"qfqday": rows}}}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        df = TencentFetcher().get_daily_data("000001", start_date="2020-01-01", end_date="2026-05-10")

    assert ",day,2020-01-01,2026-05-10,800,qfq" in captured["params"]["param"]
    assert df.empty


def test_star_market_detection_accepts_prefixed_bare_and_suffixed_codes() -> None:
    for code in ("sh688111", "SH688111", "688111", "688111.SH", "sh689009", "689009.SH"):
        assert _is_star_market(code), code
    for code in ("sh600000", "600000", "sz000001", "sz300750", "bj920199", "000688", ""):
        assert not _is_star_market(code), code


def test_lots_to_shares_keeps_star_market_volume_in_shares() -> None:
    # Tencent reports STAR Market (688/689) volume in shares already.
    assert _lots_to_shares("4421640", "sh688111") == 4421640.0
    assert _lots_to_shares("1000", "689009.SH") == 1000.0
    # Other A-shares are reported in lots (100 shares).
    assert _lots_to_shares("107450", "sh603501") == 10745000.0
    assert _lots_to_shares("147895", "sz002230") == 14789500.0
    assert _lots_to_shares("15073", "bj920199") == 1507300.0
    # Legacy call without symbol keeps the lot conversion.
    assert _lots_to_shares("12345") == 1234500.0


def test_lots_to_shares_passes_through_invalid_volume() -> None:
    assert _lots_to_shares(None, "sh688111") is None
    assert _lots_to_shares("n/a", "sh600000") == "n/a"


def test_tencent_fetcher_does_not_multiply_star_market_volume() -> None:
    payload = {
        "data": {
            "sh688111": {
                "qfqday": [
                    ["2026-09-18", "220.00", "223.61", "225.00", "219.00", "5100511", "1139410078"],
                    ["2026-09-21", "223.00", "223.59", "226.00", "221.00", "4421640", "989524266"],
                ]
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    fetcher = TencentFetcher()
    with patch("data_provider.tencent_fetcher.requests.get", lambda url, **kw: FakeResponse()):
        df = fetcher.get_daily_data("688111", start_date="2026-09-15", end_date="2026-09-22")

    assert float(df.iloc[0]["volume"]) == 5100511.0
    assert float(df.iloc[1]["volume"]) == 4421640.0
    # volume * close / amount should be ~1 when units are consistent.
    ratio = float(df.iloc[1]["volume"]) * float(df.iloc[1]["close"]) / float(df.iloc[1]["amount"])
    assert 0.99 < ratio < 1.01


def _quote_line(symbol: str, name: str, code: str, price: str, prev_close: str, amount_wan: str) -> str:
    """Build one real-format Tencent quote line (fields: 1 name, 2 code, 3 price, 4 prev close, 37 amount in 万元)."""
    fields = [""] * 50
    fields[1] = name
    fields[2] = code
    fields[3] = price
    fields[4] = prev_close
    fields[37] = amount_wan
    return 'v_' + symbol + '="' + "~".join(fields) + '";'


class _FakeQuoteResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeHttpResponse:
    """Fake requests response carrying either quote text or a JSON payload."""

    def __init__(self, text: str = "", json_data=None) -> None:
        self.text = text
        self._json_data = json_data

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_data


def test_parse_quote_fields_maps_tencent_columns() -> None:
    line = _quote_line("sz000001", "平安银行", "000001", "12.25", "12.30", "192478")
    row = _parse_quote_fields(line.split('"')[1].split("~"))

    assert row == {
        "ts_code": "000001",
        "name": "平安银行",
        "close": 12.25,
        "pre_close": 12.30,
        "amount": 1924780000.0,
    }
    assert _parse_quote_fields(["a", "b"]) is None
    bad_price = _quote_line("sz000001", "平安银行", "000001", "-", "12.30", "192478")
    assert _parse_quote_fields(bad_price.split('"')[1].split("~")) is None


def test_calc_market_stats_from_quotes_applies_a_share_limit_rules() -> None:
    df = pd.DataFrame(
        [
            {"ts_code": "000001", "name": "平安银行", "close": 13.53, "pre_close": 12.30, "amount": 100000 * 10000.0},
            {"ts_code": "600000", "name": "浦发银行", "close": 10.78, "pre_close": 10.78, "amount": 50000 * 10000.0},
            {"ts_code": "688001", "name": "华兴源创", "close": 30.00, "pre_close": 33.42, "amount": 80000 * 10000.0},
            {"ts_code": "000002", "name": "*STxxx", "close": 5.00, "pre_close": 5.26, "amount": 10000 * 10000.0},
            {"ts_code": "300001", "name": "特锐德", "close": 22.00, "pre_close": 20.00, "amount": 20000 * 10000.0},
            {"ts_code": "600001", "name": "停牌股", "close": 10.00, "pre_close": 10.00, "amount": 0.0},
        ]
    )

    stats = calc_market_stats_from_quotes(df)

    assert stats is not None
    assert stats["up_count"] == 2
    assert stats["down_count"] == 2
    assert stats["flat_count"] == 1
    assert stats["limit_up_count"] == 1
    assert stats["limit_down_count"] == 1
    assert stats["total_amount"] == pytest.approx(26.0)


def test_get_market_stats_uses_sina_list_and_tencent_quotes() -> None:
    sina_rows = [
        {"symbol": "sz000001", "code": "000001", "name": "平安银行"},
        {"symbol": "sh600000", "code": "600000", "name": "浦发银行"},
        {"symbol": "sh688001", "code": "688001", "name": "华兴源创"},
        {"symbol": "sz000002", "code": "000002", "name": "*STxxx"},
        {"symbol": "sz300001", "code": "300001", "name": "特锐德"},
    ]
    text = "\n".join(
        [
            _quote_line("sz000001", "平安银行", "000001", "13.53", "12.30", "100000"),
            _quote_line("sh600000", "浦发银行", "600000", "10.78", "10.78", "50000"),
            _quote_line("sh688001", "华兴源创", "688001", "30.00", "33.42", "80000"),
            _quote_line("sz000002", "*STxxx", "000002", "5.00", "5.26", "10000"),
            _quote_line("sz300001", "特锐德", "300001", "22.00", "20.00", "20000"),
        ]
    )

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        if "sina" in url:
            captured["sina_called"] = True
            return _FakeHttpResponse(json_data=sina_rows)
        return _FakeQuoteResponse(text)

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        stats = TencentFetcher().get_market_stats()

    assert stats is not None
    assert stats["up_count"] == 2
    assert stats["down_count"] == 2
    assert stats["flat_count"] == 1
    assert stats["limit_up_count"] == 1
    assert stats["limit_down_count"] == 1
    assert stats["total_amount"] == pytest.approx(26.0)
    assert captured["sina_called"] is True
    assert captured["url"].startswith("https://qt.gtimg.cn/q=")
    for symbol in ("sz000001", "sh600000", "sh688001", "sz000002", "sz300001"):
        assert symbol in captured["url"]


def test_get_market_stats_falls_back_to_baostock_when_sina_fails() -> None:
    codes = pd.DataFrame(
        {
            "code": ["000001", "600000", "688001", "000002", "300001"],
            "name": ["平安银行", "浦发银行", "华兴源创", "*STxxx", "特锐德"],
        }
    )
    text = "\n".join(
        [
            _quote_line("sz000001", "平安银行", "000001", "13.53", "12.30", "100000"),
            _quote_line("sh600000", "浦发银行", "600000", "10.78", "10.78", "50000"),
            _quote_line("sh688001", "华兴源创", "688001", "30.00", "33.42", "80000"),
            _quote_line("sz000002", "*STxxx", "000002", "5.00", "5.26", "10000"),
            _quote_line("sz300001", "特锐德", "300001", "22.00", "20.00", "20000"),
        ]
    )

    def fake_get(url, **kwargs):
        if "sina" in url:
            raise Exception("sina down")
        return _FakeQuoteResponse(text)

    with (
        patch("data_provider.baostock_fetcher.BaostockFetcher.get_stock_list", return_value=codes),
        patch("data_provider.tencent_fetcher.requests.get", fake_get),
    ):
        stats = TencentFetcher().get_market_stats()

    assert stats is not None
    assert stats["up_count"] == 2
    assert stats["total_amount"] == pytest.approx(26.0)


def test_get_market_stats_returns_none_when_all_stock_lists_fail() -> None:
    def failing_get(url, **kwargs):
        raise Exception("network down")

    with (
        patch("data_provider.baostock_fetcher.BaostockFetcher.get_stock_list", return_value=None),
        patch("data_provider.tushare_fetcher.TushareFetcher.get_stock_list", return_value=None),
        patch("data_provider.tencent_fetcher.requests.get", failing_get),
    ):
        assert TencentFetcher().get_market_stats() is None


def test_fetch_sina_stock_codes_paginates_and_dedupes() -> None:
    def fake_get(url, **kwargs):
        page = kwargs["params"]["page"]
        assert kwargs["params"]["num"] == 100
        assert kwargs["params"]["node"] == "hs_a"
        if page == 1:
            rows = [{"symbol": f"sh600{i:03d}", "code": f"600{i:03d}", "name": "x"} for i in range(100)]
        elif page == 2:
            rows = [{"symbol": f"sz{i:06d}", "code": f"{i:06d}", "name": "x"} for i in range(1, 101)]
        else:
            rows = [
                {"symbol": "sh600000", "code": "600000", "name": "x"},
                {"symbol": "bj920000", "code": "920000", "name": "x"},
            ]
        return _FakeHttpResponse(json_data=rows)

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        codes = TencentFetcher()._fetch_sina_stock_codes()

    assert len(codes) == 201
    assert codes.count("600000") == 1
    assert "920000" in codes


def test_fetch_sina_stock_codes_aborts_on_page_failure() -> None:
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs["params"]["page"])
        if kwargs["params"]["page"] == 2:
            raise Exception("page 2 down")
        return _FakeHttpResponse(
            json_data=[{"symbol": f"sh600{i:03d}", "code": f"600{i:03d}", "name": "x"} for i in range(100)]
        )

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        assert TencentFetcher()._fetch_sina_stock_codes() == []

    assert calls == [1, 2, 2, 2]


def test_fetch_sina_stock_codes_retries_transient_page_failure() -> None:
    attempts = []

    def fake_get(url, **kwargs):
        page = kwargs["params"]["page"]
        attempts.append(page)
        if page == 2 and attempts.count(2) < 3:
            raise Exception("page 2 flaky")
        if page == 1:
            rows = [{"symbol": f"sh600{i:03d}", "code": f"600{i:03d}", "name": "x"} for i in range(100)]
        else:
            rows = [{"symbol": "bj920000", "code": "920000", "name": "x"}]
        return _FakeHttpResponse(json_data=rows)

    with patch("data_provider.tencent_fetcher.requests.get", fake_get):
        codes = TencentFetcher()._fetch_sina_stock_codes()

    assert len(codes) == 101
    assert "920000" in codes
    assert attempts.count(2) == 3


def test_get_market_stats_returns_none_when_coverage_below_threshold() -> None:
    codes = pd.DataFrame({"code": [f"{i:06d}" for i in range(1, 11)], "name": ["x"] * 10})
    text = "\n".join(
        _quote_line(f"sz{i:06d}", "x", f"{i:06d}", "10.00", "10.00", "1000") for i in range(1, 6)
    )

    def fake_get(url, **kwargs):
        if "sina" in url:
            raise Exception("sina down")
        return _FakeQuoteResponse(text)

    with (
        patch("data_provider.baostock_fetcher.BaostockFetcher.get_stock_list", return_value=codes),
        patch("data_provider.tencent_fetcher.requests.get", fake_get),
    ):
        assert TencentFetcher().get_market_stats() is None


def test_get_market_stats_chunks_large_symbol_lists() -> None:
    codes = pd.DataFrame({"code": [f"600{i:03d}" for i in range(250)], "name": ["x"] * 250})
    calls = []

    def fake_get(url, **kwargs):
        if "sina" in url:
            raise Exception("sina down")
        calls.append(url)
        symbols = url.split("q=", 1)[1].split(",")
        text = "\n".join(_quote_line(s, "x", s[2:], "10.00", "10.00", "1000") for s in symbols)
        return _FakeQuoteResponse(text)

    with (
        patch("data_provider.baostock_fetcher.BaostockFetcher.get_stock_list", return_value=codes),
        patch("data_provider.tencent_fetcher.requests.get", fake_get),
    ):
        stats = TencentFetcher().get_market_stats()

    assert len(calls) == 3
    assert stats is not None
    assert stats["flat_count"] == 250
    assert stats["total_amount"] == pytest.approx(25.0)


def test_get_market_stats_aborts_after_repeated_chunk_failures() -> None:
    codes = pd.DataFrame({"code": [f"600{i:03d}" for i in range(250)], "name": ["x"] * 250})
    calls = []

    def failing_get(url, **kwargs):
        if "sina" in url:
            raise Exception("sina down")
        calls.append(url)
        raise Exception("boom")

    with (
        patch("data_provider.baostock_fetcher.BaostockFetcher.get_stock_list", return_value=codes),
        patch("data_provider.tencent_fetcher.requests.get", failing_get),
    ):
        assert TencentFetcher().get_market_stats() is None

    assert len(calls) == 3
