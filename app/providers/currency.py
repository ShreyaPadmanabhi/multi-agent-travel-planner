"""Currency conversion from ECB reference rates via Frankfurter (keyless).

Airlines quote fares in their own currency. Converting with a published daily
rate keeps prices comparable without inventing figures; if no rate is
available the caller leaves the USD price empty.
"""

from __future__ import annotations

from app.config import settings
from app.providers.http import ProviderError, request_json


class CurrencyProvider:
    source = "ECB reference rates (Frankfurter)"

    async def rate_to_usd(self, currency: str) -> float | None:
        """USD per one unit of `currency`, or None when no published rate exists."""
        currency = (currency or "").upper()
        if currency == "USD":
            return 1.0
        if not currency:
            return None
        try:
            payload = await request_json(
                "GET",
                settings.frankfurter_url,
                params={"base": currency, "symbols": "USD"},
                limiter="frankfurter",
            )
        except ProviderError:
            return None
        rate = (payload.get("rates") or {}).get("USD")
        return float(rate) if rate else None
