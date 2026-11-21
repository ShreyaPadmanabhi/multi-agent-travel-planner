"""Check the Duffel flight provider.

Always runs an offline check against Duffel-shaped responses. With
DUFFEL_ACCESS_TOKEN set in .env it also searches a real route.

    python scripts/test_flights.py "London" "Rome"
"""

import asyncio
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.travel import GeoLocation
from app.providers import flights as flights_module
from app.providers.flights import FlightProvider, _parse_iso_duration
from app.providers.geocoding import GeocodingProvider

LONDON = GeoLocation(name="London", display_name="London, UK", latitude=51.507, longitude=-0.128)
ROME = GeoLocation(name="Rome", display_name="Rome, Italy", latitude=41.893, longitude=12.483)

PLACES = {
    "51.507": {"data": [
        {"type": "airport", "iata_code": "LHR", "iata_city_code": "LON", "latitude": 51.47, "longitude": -0.45},
        {"type": "city", "iata_code": "LON", "latitude": 51.507, "longitude": -0.128},
    ]},
    "41.893": {"data": [
        {"type": "airport", "iata_code": "FCO", "iata_city_code": "ROM", "latitude": 41.80, "longitude": 12.25},
    ]},
}


def _segment(origin, destination, departing, arriving, number):
    return {
        "origin": {"iata_code": origin},
        "destination": {"iata_code": destination},
        "departing_at": departing,
        "arriving_at": arriving,
        "marketing_carrier": {"name": "Duffel Airways", "iata_code": "ZZ"},
        "marketing_carrier_flight_number": number,
        "passengers": [{"cabin_class_marketing_name": "Economy"}],
    }


OFFERS = {"data": {"offers": [
    {
        "total_amount": "412.30", "total_currency": "GBP", "owner": {"name": "Duffel Airways"},
        "slices": [{"duration": "P1DT1H5M", "segments": [
            _segment("LHR", "MAD", "2026-12-01T07:00:00", "2026-12-01T10:30:00", "100"),
            _segment("MAD", "FCO", "2026-12-02T06:00:00", "2026-12-02T08:05:00", "101"),
        ]}],
    },
    {
        "total_amount": "150.00", "total_currency": "USD", "owner": {"name": "Duffel Airways"},
        "slices": [{"duration": "PT2H35M", "segments": [
            _segment("LHR", "FCO", "2026-12-01T09:00:00", "2026-12-01T12:35:00", "200"),
        ]}],
    },
]}}


async def offline_check() -> None:
    seen: dict = {}

    async def fake_request(method, url, *, params=None, json_body=None, headers=None, **_):
        if url.endswith("/places/suggestions"):
            return PLACES[str(params["lat"])]
        seen["body"], seen["headers"] = json_body, headers
        return OFFERS

    class FixedRates:
        async def rate_to_usd(self, currency):
            return {"USD": 1.0, "GBP": 1.25}.get(currency)

    original_request, original_token = flights_module.request_json, settings.duffel_access_token
    flights_module.request_json = fake_request
    settings.duffel_access_token = "duffel_test_offline"
    try:
        options = await FlightProvider(currency=FixedRates()).search(LONDON, ROME, date(2026, 12, 1), travelers=2)
    finally:
        flights_module.request_json, settings.duffel_access_token = original_request, original_token

    slice_ = seen["body"]["data"]["slices"][0]
    assert (slice_["origin"], slice_["destination"]) == ("LON", "ROM"), slice_
    assert len(seen["body"]["data"]["passengers"]) == 2
    assert seen["headers"]["Duffel-Version"] == "v2"
    assert [o.price_usd for o in options] == [150.0, 515.38], [o.price_usd for o in options]
    assert options[1].duration_minutes == 1505 and options[1].stops == 1
    assert "412.30 GBP" in options[1].booking_class
    assert options[0].source == "Duffel (test mode)"
    assert _parse_iso_duration("PT45M") == 45 and _parse_iso_duration("") is None
    print("offline check OK:", [(o.provider, o.price_usd, o.duration_label, o.stops) for o in options])


async def live_check(origin: str, destination: str) -> None:
    if not settings.flights_enabled:
        print("live check skipped: set DUFFEL_ACCESS_TOKEN in .env to search a real route")
        return
    geocoder = GeocodingProvider()
    a, b = await geocoder.geocode(origin), await geocoder.geocode(destination)
    provider = FlightProvider()
    print(f"airports: {await provider.nearest_airport_code(a)} -> {await provider.nearest_airport_code(b)}")
    options = await provider.search(a, b, date.today() + timedelta(days=30), travelers=1)
    print(f"{len(options)} offers ({settings.flight_mode} mode)")
    for option in options[:5]:
        price = f"${option.price_usd:,.2f}" if option.price_usd is not None else "no USD price"
        print(f"  {option.provider:24} {option.duration_label:9} stops={option.stops} {price} {option.booking_class}")


async def main() -> None:
    await offline_check()
    await live_check(sys.argv[1] if len(sys.argv) > 1 else "London", sys.argv[2] if len(sys.argv) > 2 else "Rome")


asyncio.run(main())
