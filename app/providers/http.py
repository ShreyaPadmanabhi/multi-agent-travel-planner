"""Shared async HTTP plumbing: connection reuse, TTL caching, retries, rate limiting."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when an upstream data provider cannot satisfy a request."""


class RateLimiter:
    """Serialises calls to a host and enforces a minimum gap between them.

    Nominatim and the public Overpass instances are volunteer-run and cap
    clients at roughly one request per second; exceeding that gets you blocked.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


class TTLCache:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)


_cache = TTLCache(settings.cache_ttl_seconds)
_limiters: dict[str, RateLimiter] = {
    "nominatim": RateLimiter(1.1),
    "overpass": RateLimiter(1.1),
    "wikidata": RateLimiter(1.0),
    "wikipedia": RateLimiter(0.2),
    "osrm": RateLimiter(0.3),
    "open-meteo": RateLimiter(0.2),
    "duffel": RateLimiter(0.2),
    "frankfurter": RateLimiter(0.2),
}


def _cache_key(method: str, url: str, params: Any, body: Any) -> str:
    return json.dumps([method, url, params, body], sort_keys=True, default=str)


async def request_json(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: Any = None,
    json_body: Any = None,
    headers: dict | None = None,
    limiter: str | None = None,
    use_cache: bool = True,
    retries: int = 2,
) -> Any:
    """Perform an HTTP request and decode JSON, with caching and bounded retries."""
    key = _cache_key(method, url, params, data or json_body)
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    request_headers = {"User-Agent": settings.user_agent, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if limiter and limiter in _limiters:
            await _limiters[limiter].acquire()
        try:
            async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    json=json_body,
                    headers=request_headers,
                )
            # Back off and retry on throttling or transient server errors.
            if response.status_code in (429, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"upstream returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            payload = response.json()
            if use_cache:
                _cache.set(key, payload)
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))

    raise ProviderError(f"{method} {url} failed after {retries + 1} attempts: {last_error}")
