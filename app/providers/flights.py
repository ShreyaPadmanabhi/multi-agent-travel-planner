"""Flight offers from the Duffel API.

Duffel has a free test mode: a `duffel_test_...` token returns real-shaped
offers (including the fictional "Duffel Airways") at no cost. A live token
returns bookable airline inventory.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from app.config import settings
from app.models.travel import FlightSegment, GeoLocation, TravelOption
from app.providers.currency import CurrencyProvider
from app.providers.http import ProviderError, request_json
from app.providers.routing import haversine_km

# How far from a city to look for its airport.
AIRPORT_SEARCH_RADIUS_M = 100_000


def _parse_iso_duration(value: str) -> int | None:
    """Convert an ISO-8601 duration such as 'PT7H35M' or 'P1DT2H' into total minutes."""
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", value or "")
    if not match or not any(match.groups()):
        return None
    days, hours, minutes = (int(group or 0) for group in match.groups())
    return days * 1440 + hours * 60 + minutes


class FlightProvider:
    source = "Duffel"

    def __init__(self, currency: CurrencyProvider | None = None) -> None:
        self.currency = currency or CurrencyProvider()

    @property
    def enabled(self) -> bool:
        return settings.flights_enabled

    @property
    def test_mode(self) -> bool:
        return settings.duffel_access_token.startswith("duffel_test_")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.duffel_access_token}",
            "Duffel-Version": "v2",
            "Content-Type": "application/json",
        }

    async def nearest_airport_code(self, location: GeoLocation) -> str | None:
        """City IATA code where one exists (LON covers every London airport), else the airport's."""
        payload = await request_json(
            "GET",
            f"{settings.duffel_base_url}/places/suggestions",
            params={
                "lat": round(location.latitude, 4),
                "lng": round(location.longitude, 4),
                "rad": AIRPORT_SEARCH_RADIUS_M,
            },
            headers=self._headers(),
            limiter="duffel",
        )
        places = [p for p in payload.get("data") or [] if p.get("iata_code")]
        if not places:
            return None

        def distance(place: dict) -> float:
            if place.get("latitude") is None or place.get("longitude") is None:
                return 1e9
            return haversine_km(location.latitude, location.longitude, place["latitude"], place["longitude"])

        nearest = min(places, key=distance)
        if nearest.get("type") == "city":
            return nearest["iata_code"]
        return nearest.get("iata_city_code") or nearest["iata_code"]

    async def search(
        self,
        origin: GeoLocation,
        destination: GeoLocation,
        departure_date: date,
        travelers: int,
        max_results: int = 8,
    ) -> list[TravelOption]:
        if not self.enabled:
            raise ProviderError("Duffel access token is not configured.")

        origin_code = await self.nearest_airport_code(origin)
        destination_code = await self.nearest_airport_code(destination)
        if not origin_code or not destination_code:
            raise ProviderError("Could not resolve airports for the requested route.")
        if origin_code == destination_code:
            return []

        payload = await request_json(
            "POST",
            f"{settings.duffel_base_url}/air/offer_requests",
            params={"return_offers": "true", "supplier_timeout": 20000},
            json_body={
                "data": {
                    "slices": [
                        {
                            "origin": origin_code,
                            "destination": destination_code,
                            "departure_date": departure_date.isoformat(),
                        }
                    ],
                    "passengers": [{"type": "adult"} for _ in range(travelers)],
                    "cabin_class": "economy",
                }
            },
            headers=self._headers(),
            limiter="duffel",
            # Offers expire within minutes, so a cached search would show stale fares.
            use_cache=False,
        )

        offers = sorted(
            (payload.get("data") or {}).get("offers") or [],
            key=lambda o: float(o.get("total_amount") or 1e12),
        )
        options: list[TravelOption] = []
        rates: dict[str, float | None] = {}

        for offer in offers:
            slices = offer.get("slices") or []
            raw_segments = (slices[0].get("segments") or []) if slices else []
            if not raw_segments:
                continue

            segments = [
                FlightSegment(
                    carrier=(seg.get("marketing_carrier") or {}).get("name", ""),
                    flight_number=f"{(seg.get('marketing_carrier') or {}).get('iata_code', '')}"
                    f"{seg.get('marketing_carrier_flight_number', '')}",
                    origin=(seg.get("origin") or {}).get("iata_code", ""),
                    destination=(seg.get("destination") or {}).get("iata_code", ""),
                    departure_time=datetime.fromisoformat(seg["departing_at"]),
                    arrival_time=datetime.fromisoformat(seg["arriving_at"]),
                )
                for seg in raw_segments
            ]

            # Convert once per currency rather than per offer.
            currency = (offer.get("total_currency") or "").upper()
            amount = float(offer.get("total_amount") or 0)
            if currency not in rates:
                rates[currency] = await self.currency.rate_to_usd(currency)
            rate = rates[currency]
            price_usd = round(amount * rate, 2) if rate and amount else None

            cabin = ((raw_segments[0].get("passengers") or [{}])[0]).get("cabin_class_marketing_name", "")
            note = "" if currency == "USD" else f" (quoted {amount:,.2f} {currency})"
            options.append(
                TravelOption(
                    mode="flight",
                    provider=(offer.get("owner") or {}).get("name", "Airline"),
                    origin=segments[0].origin,
                    destination=segments[-1].destination,
                    departure_time=segments[0].departure_time,
                    arrival_time=segments[-1].arrival_time,
                    duration_minutes=_parse_iso_duration(slices[0].get("duration") or ""),
                    price_usd=price_usd,
                    stops=len(segments) - 1,
                    booking_class=f"{cabin}{note}".strip(),
                    segments=segments,
                    source=f"{self.source} (test mode)" if self.test_mode else self.source,
                )
            )
            if len(options) >= max_results:
                break

        return options
