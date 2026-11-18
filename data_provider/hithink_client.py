# -*- coding: utf-8 -*-
"""Shared HTTP client for Hithink (同花顺) Financial-API.

One key, one base URL, one envelope contract. Shared by HithinkFetcher
(market breadth) and HithinkFundamentalAdapter (structured financials)
so the auth / error handling is not duplicated per caller.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

HITHINK_BASE_URL = "https://fuyao.aicubes.cn"
_HITHINK_TIMEOUT_SECONDS = 10


class HithinkApiError(Exception):
    """Hithink API returned an error envelope or an HTTP error."""


class HithinkRateLimitError(HithinkApiError):
    """Hithink dynamic rate limiting hit (HTTP 429 or envelope code 4001)."""


def to_thscode(code: str) -> Optional[str]:
    """6-digit code -> full thscode, e.g. ``600519`` -> ``600519.SH``."""
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    if len(digits) != 6:
        return None
    if digits.startswith(("60", "68", "90")):
        suffix = "SH"
    elif digits.startswith(("00", "30", "20")):
        suffix = "SZ"
    elif digits.startswith(("43", "92", "87")):
        suffix = "BJ"
    else:
        return None
    return f"{digits}.{suffix}"


def from_thscode(thscode: str) -> str:
    """Full thscode -> 6-digit code, e.g. ``600519.SH`` -> ``600519``."""
    return str(thscode or "").split(".")[0]


class HithinkClient:
    """Minimal Hithink REST client: ``X-api-key`` header, envelope check."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = _HITHINK_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("HITHINK_FINANCE_API_KEY") or "").strip()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        """GET one endpoint; returns the envelope's ``data`` payload.

        Raises :class:`HithinkRateLimitError` on HTTP 429 / envelope code
        4001, :class:`HithinkApiError` on any other non-zero envelope code
        or HTTP error. The key is never logged.
        """
        if not self.api_key:
            raise HithinkApiError("Hithink API key is not configured")
        url = HITHINK_BASE_URL + path
        try:
            response = requests.get(
                url,
                params=params or {},
                headers={"X-api-key": self.api_key, "Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise HithinkApiError(f"Hithink request failed: {exc}") from exc
        if response.status_code == 429:
            raise HithinkRateLimitError(f"Hithink rate limited (HTTP 429): {path}")
        if response.status_code != 200:
            raise HithinkApiError(f"Hithink HTTP {response.status_code}: {path}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HithinkApiError(f"Hithink non-JSON response: {path}") from exc
        code = payload.get("code")
        if code == 4001:
            raise HithinkRateLimitError(f"Hithink rate limited (code 4001): {path}")
        if code != 0:
            raise HithinkApiError(
                f"Hithink error code={code} message={payload.get('message')}: {path}"
            )
        return payload.get("data")
