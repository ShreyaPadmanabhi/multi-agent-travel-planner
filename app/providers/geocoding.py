"""City and address resolution via OpenStreetMap Nominatim (no API key required)."""

from __future__ import annotations

from app.config import settings
from app.models.travel import GeoLocation
from app.providers.http import ProviderError, request_json


class GeocodingProvider:
    source = "OpenStreetMap Nominatim"

    async def geocode(self, query: str) -> GeoLocation:
        cleaned = query.strip()
        if not cleaned:
            raise ProviderError("Empty location query.")

        payload = await request_json(
            "GET",
            f"{settings.nominatim_url}/search",
            params={
                "q": cleaned,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            },
            limiter="nominatim",
        )

        if not payload:
            raise ProviderError(f"Could not find a location matching {cleaned!r}.")

        top = payload[0]
        address = top.get("address", {}) or {}
        return GeoLocation(
            name=top.get("name") or cleaned,
            display_name=top.get("display_name", cleaned),
            latitude=float(top["lat"]),
            longitude=float(top["lon"]),
            country_code=(address.get("country_code") or "").upper(),
        )
