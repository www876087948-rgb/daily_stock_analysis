# -*- coding: utf-8 -*-
"""Tencent direct daily K-line fetcher for A-share fallback routing."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

try:
    import exchange_calendars as xcals
except ImportError:  # pragma: no cover - dependency is present in supported installs
    xcals = None

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, calc_market_stats_from_quotes, normalize_stock_code, is_bse_code

logger = logging.getLogger(__name__)

_MAX_KLINE_BARS = 800

# 市场统计（get_market_stats）批量行情参数：
# - 每批最多 100 只（实测 100 只约 1.5 秒，腾讯无显式批次上限）；
# - 覆盖率低于 80% 视为数据不可信，返回 None 继续降级；
# - 连续 3 批失败直接放弃，避免单源故障拖垮整轮分析。
_QUOTE_BATCH_SIZE = 100
_MIN_MARKET_COVERAGE = 0.8
_MAX_CONSECUTIVE_CHUNK_FAILURES = 3

# 全市场代码清单（新浪行情中心）：单页最多 100 只，实测全市场约 5500 只。
_SINA_LIST_ENDPOINT = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_SINA_LIST_PAGE_SIZE = 100
_SINA_LIST_MAX_PAGES = 80
_SINA_LIST_TIMEOUT_SECONDS = 15
_SINA_LIST_PAGE_RETRIES = 3


class TencentFetcher(BaseFetcher):
    """Fetch qfq daily K-line data from Tencent's direct quote endpoint."""

    name = "TencentFetcher"
    # This direct endpoint is the last-resort A-share daily fallback. Keeping
    # it at priority 0 made a single Efinance failure skip the richer built-in
    # fallback chain and try Tencent before AkShare/PyTDX/Baostock/YFinance.
    priority = 5
    allow_empty_daily_data = True

    _KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    _QUOTE_ENDPOINT = "https://qt.gtimg.cn/q"
    _HTTP_TIMEOUT_SECONDS = 8

    def __init__(self) -> None:
        self.priority = _read_tencent_priority()

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = _to_tencent_symbol(stock_code)
        if not symbol:
            raise DataFetchError(f"TencentFetcher unsupported stock code: {stock_code}")

        lookback = _estimate_lookback_days(start_date=start_date, end_date=end_date)
        explicit_start = _format_tencent_date(start_date)
        explicit_end = _format_tencent_date(end_date)
        explicit_window = (
            f"{explicit_start},{explicit_end}"
            if explicit_start and explicit_end
            else ","
        )
        response = requests.get(
            self._KLINE_ENDPOINT,
            params={"param": f"{symbol},day,{explicit_window},{lookback},qfq"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_kline_rows(payload, symbol=symbol)
        if not rows:
            logger.info("TencentFetcher empty daily history for %s", stock_code)
            return _empty_daily_frame()

        df = pd.DataFrame(rows)
        first_returned_date = _first_returned_date(df)
        if first_returned_date and _is_capped_history_incomplete(
            first_returned_date=first_returned_date,
            start_date=start_date,
            lookback=lookback,
            returned_rows=len(rows),
        ):
            logger.info(
                "TencentFetcher incomplete capped daily history for %s: first_date=%s requested_start=%s",
                stock_code,
                first_returned_date,
                start_date,
            )
            return _empty_daily_frame()

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            logger.info(
                "TencentFetcher daily history outside requested range for %s: %s~%s",
                stock_code,
                start_date,
                end_date,
            )
            return _empty_daily_frame()
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "pct_chg" not in normalized.columns:
            normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100
        normalized = normalized[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        return normalized

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        symbol = _to_tencent_symbol(stock_code)
        if not symbol:
            return None
        try:
            response = requests.get(
                f"{self._QUOTE_ENDPOINT}={symbol}",
                headers={"Referer": "https://finance.qq.com", "User-Agent": "Mozilla/5.0"},
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = "gbk"
            content = response.text.strip()
            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end <= data_start:
                return None
            fields = content[data_start + 1:data_end].split("~")
            if len(fields) <= 2 or fields[2].strip() != symbol[2:]:
                return None
            name = fields[1].strip()
            return name or None
        except Exception:
            logger.debug(
                "TencentFetcher stock name lookup failed for %s",
                stock_code,
                exc_info=True,
            )
            return None

    def get_market_stats(self) -> Optional[dict]:
        """获取市场涨跌统计（腾讯全市场行情兜底）。

        东财（efinance/akshare）与 Tushare 均不可用时，由本方法接手：
        以新浪/Baostock/Tushare 列表（不走东财）取全市场 A 股代码，分批拉取腾讯
        行情后按 A 股涨跌停规则计算涨跌家数、涨跌停与总成交额。
        覆盖不足或任一环节失败返回 None，由调用方继续降级。
        """
        codes = self._get_all_a_share_codes()
        if not codes:
            return None

        symbols = []
        for code in codes:
            symbol = _to_tencent_symbol(code)
            if symbol:
                symbols.append(symbol)
        if not symbols:
            return None

        rows: list[dict[str, Any]] = []
        consecutive_failures = 0
        for start in range(0, len(symbols), _QUOTE_BATCH_SIZE):
            chunk = symbols[start:start + _QUOTE_BATCH_SIZE]
            try:
                chunk_rows = self._fetch_quote_chunk(chunk)
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "[Tencent] market stats quote chunk failed (%d/%d consecutive)",
                    consecutive_failures,
                    _MAX_CONSECUTIVE_CHUNK_FAILURES,
                    exc_info=True,
                )
                if consecutive_failures >= _MAX_CONSECUTIVE_CHUNK_FAILURES:
                    logger.warning("[Tencent] market stats aborted: too many chunk failures")
                    return None
                continue
            consecutive_failures = 0
            rows.extend(chunk_rows)

        if not rows:
            logger.warning("[Tencent] market stats got no quotes")
            return None

        coverage = len(rows) / len(symbols)
        logger.info(
            "[Tencent] market stats coverage %.1f%% (%d/%d symbols)",
            coverage * 100,
            len(rows),
            len(symbols),
        )
        if coverage < _MIN_MARKET_COVERAGE:
            logger.warning(
                "[Tencent] market stats coverage %.1f%% below %.0f%% threshold; discarding",
                coverage * 100,
                _MIN_MARKET_COVERAGE * 100,
            )
            return None

        df = pd.DataFrame(rows, columns=["ts_code", "name", "close", "pre_close", "amount"])
        return calc_market_stats_from_quotes(df)

    def _get_all_a_share_codes(self) -> list[str]:
        """全市场 A 股代码清单：新浪 → Baostock → Tushare，首个非空者胜。

        清单源均不走东财。Baostock SDK 在云端机房 IP 上登录经常挂起，
        故首选新浪 HTTPS 分页接口（已实测覆盖全市场约 5500 只）。
        """
        for source, getter in (
            ("Sina", self._fetch_sina_stock_codes),
            ("Baostock", self._fetch_baostock_stock_codes),
            ("Tushare", self._fetch_tushare_stock_codes),
        ):
            try:
                codes = getter()
            except Exception:
                logger.warning("[Tencent] %s stock list failed; trying next source", source, exc_info=True)
                continue
            if codes:
                logger.info("[Tencent] stock list from %s: %d codes", source, len(codes))
                return codes
            logger.info("[Tencent] %s stock list empty; trying next source", source)
        return []

    def _fetch_sina_stock_codes(self) -> list[str]:
        """全市场 A 股代码清单（新浪行情中心，HTTPS 分页）。

        每页最多重试 3 次；某页 3 次都失败才放弃本源，交给下一源，
        避免单次网络抖动丢掉整份清单，也避免缺页导致宽度失真。
        """
        codes: list[str] = []
        for page in range(1, _SINA_LIST_MAX_PAGES + 1):
            rows = self._fetch_sina_list_page(page)
            if rows is None:
                logger.warning("[Tencent] Sina stock list page %d failed after retries; giving up Sina source", page)
                return []
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = normalize_stock_code(str(row.get("code") or row.get("symbol") or ""))
                if code and len(code) == 6 and code.isdigit():
                    codes.append(code)
            if len(rows) < _SINA_LIST_PAGE_SIZE:
                break
        return list(dict.fromkeys(codes))

    def _fetch_sina_list_page(self, page: int) -> list | None:
        """拉一页新浪清单；3 次重试都失败返回 None。"""
        last_error: Exception | None = None
        for _attempt in range(_SINA_LIST_PAGE_RETRIES):
            try:
                response = requests.get(
                    _SINA_LIST_ENDPOINT,
                    params={
                        "page": page,
                        "num": _SINA_LIST_PAGE_SIZE,
                        "sort": "symbol",
                        "asc": 1,
                        "node": "hs_a",
                        "_s_r_a": "page",
                    },
                    headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
                    timeout=_SINA_LIST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 - transient network errors are retried
                last_error = exc
        logger.warning(
            "[Tencent] Sina stock list page %d failed after %d attempts",
            page,
            _SINA_LIST_PAGE_RETRIES,
            exc_info=last_error,
        )
        return None

    def _fetch_baostock_stock_codes(self) -> list[str]:
        """全市场 A 股代码清单（Baostock，不走东财）。延迟导入，未安装时返回空清单。"""
        try:
            from .baostock_fetcher import BaostockFetcher
        except Exception:
            logger.warning("[Tencent] BaostockFetcher unavailable; market stats skipped", exc_info=True)
            return []
        try:
            df = BaostockFetcher().get_stock_list()
        except Exception:
            logger.warning("[Tencent] Baostock stock list failed; market stats skipped", exc_info=True)
            return []
        if df is None or df.empty or "code" not in df.columns:
            return []
        codes: list[str] = []
        for raw in df["code"].astype(str):
            code = normalize_stock_code(raw)
            if code and len(code) == 6 and code.isdigit():
                codes.append(code)
        return list(dict.fromkeys(codes))

    @staticmethod
    def _fetch_tushare_stock_codes() -> list[str]:
        """全市场 A 股代码清单（Tushare stock_basic，需配置 Token；未配置时返回空清单）。"""
        try:
            from .tushare_fetcher import TushareFetcher
        except Exception:
            return []
        try:
            df = TushareFetcher().get_stock_list()
        except Exception:
            logger.warning("[Tencent] Tushare stock list failed", exc_info=True)
            return []
        if df is None or df.empty or "code" not in df.columns:
            return []
        codes: list[str] = []
        for raw in df["code"].astype(str):
            code = normalize_stock_code(raw)
            if code and len(code) == 6 and code.isdigit():
                codes.append(code)
        return list(dict.fromkeys(codes))

    def _fetch_quote_chunk(self, symbols: list[str]) -> list[dict[str, Any]]:
        """批量拉取一组腾讯行情并解析为统计行。"""
        url = f"{self._QUOTE_ENDPOINT}={','.join(symbols)}"
        response = requests.get(
            url,
            headers={"Referer": "https://finance.qq.com", "User-Agent": "Mozilla/5.0"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        rows: list[dict[str, Any]] = []
        for match in _QUOTE_LINE_RE.finditer(response.text or ""):
            row = _parse_quote_fields(match.group(1).split("~"))
            if row is not None:
                rows.append(row)
        return rows


_QUOTE_LINE_RE = re.compile(r'v_(?:sh|sz|bj)\d+="([^"]*)";')


def _parse_quote_fields(fields: list[str]) -> Optional[dict[str, Any]]:
    """解析一行腾讯行情为统计行。

    字段：1 名称、2 代码、3 现价、4 昨收、37 成交额（万元）。
    成交额统一折算为元，与其他数据源的口径一致。
    """
    if len(fields) <= 37:
        return None
    code = (fields[2] or "").strip()
    name = (fields[1] or "").strip()
    try:
        close = float(fields[3])
        pre_close = float(fields[4])
        amount_wan = float(fields[37])
    except (TypeError, ValueError):
        return None
    if not code or not name:
        return None
    return {
        "ts_code": code,
        "name": name,
        "close": close,
        "pre_close": pre_close,
        "amount": amount_wan * 10000.0,
    }


def _to_tencent_symbol(stock_code: str) -> str:
    raw_code = (stock_code or "").strip().upper()
    code = normalize_stock_code(stock_code)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if raw_code.startswith(("SH", "SS")) or raw_code.endswith((".SH", ".SS")):
        return f"sh{code}"
    if raw_code.startswith("SZ") or raw_code.endswith(".SZ"):
        return f"sz{code}"
    if raw_code.startswith("BJ") or raw_code.endswith(".BJ"):
        return f"bj{code}"
    if is_bse_code(code):
        return f"bj{code}"
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _read_tencent_priority() -> int:
    raw_value = os.getenv("TENCENT_PRIORITY", "5")
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "TENCENT_PRIORITY=%r is not a valid integer; falling back to 5",
            raw_value,
        )
        return 5


def _estimate_lookback_days(*, start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days = max(1, (end - start).days + 1)
    except ValueError:
        calendar_days = 90
    # Trading days are sparse over calendar days; add margin for holidays/suspensions.
    return max(30, min(_MAX_KLINE_BARS, int(calendar_days * 1.8) + 20))


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _first_returned_date(df: pd.DataFrame) -> Optional[str]:
    if "date" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%Y-%m-%d")


def _is_capped_history_incomplete(
    *,
    first_returned_date: str,
    start_date: str,
    lookback: int,
    returned_rows: int,
) -> bool:
    hit_cap = lookback >= _MAX_KLINE_BARS and returned_rows >= _MAX_KLINE_BARS
    if not hit_cap:
        return False
    try:
        first = datetime.strptime(first_returned_date, "%Y-%m-%d")
        requested_start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return False
    return first > _first_trading_date_on_or_after(requested_start)


def _first_trading_date_on_or_after(start_date: datetime) -> datetime:
    if xcals is not None:
        try:
            cal = xcals.get_calendar("XSHG")
            session = cal.date_to_session(start_date.date(), direction="next")
            return datetime.combine(session.date(), datetime.min.time())
        except Exception:
            pass

    current = start_date
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _format_tencent_date(date_text: str) -> Optional[str]:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _is_star_market(symbol: str) -> bool:
    """Return True for STAR Market codes (688xxx, and 689xxx CDRs).

    Accepts Tencent symbols (``sh688111``), bare codes (``688111``) and
    suffixed codes (``688111.SH``).
    """
    code = (symbol or "").strip().lower()
    if code[:2] in ("sh", "sz", "bj"):
        code = code[2:]
    return code.startswith(("688", "689"))


def _lots_to_shares(volume: Any, symbol: str = "") -> Any:
    """Convert Tencent daily K-line volume to shares.

    Tencent reports volume in lots (手, 100 shares) for most A-shares, but in
    shares (股) for STAR Market stocks (688/689). Multiplying STAR Market volume
    by 100 inflates it 100x (Refs #2350).
    """
    try:
        value = float(volume)
    except (TypeError, ValueError):
        return volume
    return value if _is_star_market(symbol) else value * 100


def _extract_kline_rows(payload: dict[str, Any], *, symbol: str) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return []
    rows = item.get("qfqday") or item.get("day") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        amount: Optional[Any] = row[6] if len(row) > 6 else None
        result.append(
            {
                "date": str(row[0]),
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": _lots_to_shares(row[5], symbol),
                "amount": amount,
            }
        )
    return result
