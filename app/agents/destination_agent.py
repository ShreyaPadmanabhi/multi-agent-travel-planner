import logging

from app.agents.base import BaseAgent, normalise_name
from app.config import settings
from app.models.travel import GeoLocation, Place, TravelPlan
from app.providers.geocoding import GeocodingProvider
from app.providers.http import ProviderError
from app.providers.places import PlacesProvider
from app.providers.routing import haversine_km

logger = logging.getLogger(__name__)

# Nominatim allows one request per second, so cap how many requested places
# are resolved individually.
MAX_MUST_VISIT = 10
# A requested place geocoded this close to a discovered one is the same place.
SAME_PLACE_KM = 0.2


class DestinationAgent(BaseAgent):
    """Resolves the origin and destination, then discovers real attractions."""

    name = "DestinationAgent"
    description = "Geocodes the trip and finds notable places to visit."
    source = "Nominatim + Wikidata + Wikipedia"

    def __init__(
        self,
        geocoder: GeocodingProvider | None = None,
        places: PlacesProvider | None = None,
    ) -> None:
        self.geocoder = geocoder or GeocodingProvider()
        self.places = places or PlacesProvider()

    async def run(self, plan: TravelPlan) -> TravelPlan:
        request = plan.request

        destination = await self.geocoder.geocode(request.destination)
        plan.destination_location = destination
        plan.destination_label = destination.display_name

        # Imported trip material often has no origin; transport then skips itself.
        if request.origin.strip():
            plan.origin_location = await self.geocoder.geocode(request.origin)

        # Allow roughly four candidate stops per day so the itinerary agent has
        # slack to drop places that do not fit the schedule.
        plan.places = await self.places.find_places(
            destination, request.interests, limit=max(request.days * 4, 8)
        )
        if request.must_visit:
            await self._pin_must_visit(plan, destination)
        return plan

    async def _pin_must_visit(self, plan: TravelPlan, destination: GeoLocation) -> None:
        """Resolve places the traveller asked for and put them at the front of the list."""
        by_name = {normalise_name(p.name): p for p in plan.places}
        pinned: list[Place] = []
        unresolved: list[str] = []

        for name in plan.request.must_visit[:MAX_MUST_VISIT]:
            existing = by_name.get(normalise_name(name))
            if existing:
                existing.pinned = True
                pinned.append(existing)
                continue
            try:
                location = await self.geocoder.geocode(f"{name}, {plan.request.destination}")
            except ProviderError:
                unresolved.append(name)
                continue
            distance = haversine_km(*destination.coordinates, *location.coordinates)
            if distance > settings.must_visit_radius_km:
                unresolved.append(name)
                continue
            # Same spot under another name ("Colosseo" vs "Colosseum"): pin the
            # discovered entry, which carries the description and notability.
            nearby = [
                p for p in plan.places
                if haversine_km(p.latitude, p.longitude, *location.coordinates) <= SAME_PLACE_KM
            ]
            if nearby:
                match = min(nearby, key=lambda p: haversine_km(p.latitude, p.longitude, *location.coordinates))
                if not match.pinned:
                    match.pinned = True
                    pinned.append(match)
                continue
            pinned.append(
                Place(
                    name=name,
                    category="attraction",
                    description="Requested in your trip material.",
                    address=location.display_name,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    pinned=True,
                    source=self.geocoder.source,
                )
            )

        if unresolved:
            plan.warnings.append(
                f"Could not place {len(unresolved)} requested stop(s) near {destination.short_name}: "
                + ", ".join(unresolved)
            )
        pinned_ids = {id(p) for p in pinned}
        plan.places = pinned + [p for p in plan.places if id(p) not in pinned_ids]

    def success_message(self, plan: TravelPlan) -> str:
        pinned = sum(1 for p in plan.places if p.pinned)
        suffix = f", {pinned} requested by you" if pinned else ""
        return f"Found {len(plan.places)} notable places near {plan.destination_label.split(',')[0]}{suffix}."
