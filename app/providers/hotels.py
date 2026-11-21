"""Hotel listings from OpenStreetMap (keyless): real, mapped properties without rates.

No free API offers priced hotel inventory any more, so the plan shows real
properties and the booking agent links to a prefilled search for live rates.
"""

from __future__ import annotations

from datetime import date

from app.config import settings
from app.models.travel import GeoLocation, HotelOption
from app.providers.http import request_json
from app.providers.routing import haversine_km


class HotelProvider:
    listing_source = "OpenStreetMap"

    async def search(
        self,
        location: GeoLocation,
        check_in: date,
        check_out: date,
        travelers: int,
        limit: int = 15,
    ) -> list[HotelOption]:
        return await self._search_openstreetmap(location, limit)

    async def _search_openstreetmap(self, location: GeoLocation, limit: int) -> list[HotelOption]:
        """Real, mapped accommodation near the destination. Names and locations only, no rates."""
        query = (
            f"[out:json][timeout:40];\n"
            f'nwr["tourism"~"^(hotel|guest_house|hostel|apartment)$"]["name"]'
            f"(around:{settings.hotel_search_radius_km * 1000},{location.latitude},{location.longitude});\n"
            f"out center tags 120;"
        )

        payload = await request_json(
            "POST",
            settings.overpass_url,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            limiter="overpass",
        )

        hotels: list[HotelOption] = []
        for element in payload.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name")
            if not name:
                continue

            if "lat" in element:
                latitude, longitude = float(element["lat"]), float(element["lon"])
            elif element.get("center"):
                latitude, longitude = float(element["center"]["lat"]), float(element["center"]["lon"])
            else:
                continue

            stars_tag = tags.get("stars", "")
            address_parts = [
                " ".join(filter(None, [tags.get("addr:housenumber"), tags.get("addr:street")])),
                tags.get("addr:city"),
            ]

            amenities = [
                label
                for tag, label in (
                    ("internet_access", "wifi"),
                    ("swimming_pool", "pool"),
                    ("air_conditioning", "air conditioning"),
                    ("breakfast", "breakfast"),
                )
                if tags.get(tag) not in (None, "no")
            ]

            hotels.append(
                HotelOption(
                    name=name,
                    stars=int(stars_tag) if stars_tag.isdigit() and 1 <= int(stars_tag) <= 5 else None,
                    address=", ".join(p for p in address_parts if p),
                    latitude=latitude,
                    longitude=longitude,
                    amenities=amenities,
                    distance_to_center_km=round(
                        haversine_km(location.latitude, location.longitude, latitude, longitude), 1
                    ),
                    website=tags.get("website") or tags.get("contact:website", ""),
                    source=self.listing_source,
                )
            )

        # Without rates, proximity to the centre is the most useful ranking signal.
        hotels.sort(key=lambda h: (h.distance_to_center_km or 1e9, h.name))
        return hotels[:limit]
