"""Turns an approved plan into booking hand-offs.

This agent deliberately stops short of taking payment: real booking needs
payment processing, inventory holds and idempotency against double-charging,
none of which can be demonstrated honestly against a test API. It produces
provider links prefilled with the approved route, dates and party size, so the
traveller completes checkout on the provider's own site.
"""

from __future__ import annotations

from urllib.parse import quote_plus, urlencode

from app.agents.base import AgentSkipped, BaseAgent, normalise_name
from app.models.travel import BookingItem, HotelOption, TravelPlan


class BookingAgent(BaseAgent):
    name = "BookingAgent"
    description = "Prepares prefilled booking links for the approved transport, stay and tickets."
    source = "Provider deep links (no payment taken)"

    async def run(self, plan: TravelPlan) -> TravelPlan:
        if plan.status != "approved":
            raise AgentSkipped("Booking only runs on a plan the traveller has approved.")

        items: list[BookingItem] = []
        if plan.recommended_travel:
            items.append(self._transport_item(plan))
        items.extend(self._hotel_items(plan))
        items.extend(self._ticket_items(plan))

        if not items:
            raise AgentSkipped("Nothing in the approved plan can be booked.")
        plan.bookings = items
        return plan

    @staticmethod
    def _transport_item(plan: TravelPlan) -> BookingItem:
        travel = plan.recommended_travel
        request = plan.request
        origin = travel.origin or request.origin
        destination = travel.destination or request.destination

        if travel.mode == "flight":
            query = f"Flights from {origin} to {destination} on {request.start_date} for {request.travelers} adults"
            url = travel.booking_url or f"https://www.google.com/travel/flights?q={quote_plus(query)}"
            provider = travel.provider
        else:
            url = "https://www.google.com/maps/dir/?" + urlencode(
                {"api": 1, "origin": origin, "destination": destination, "travelmode": "driving"}
            )
            provider = "Google Maps directions"

        return BookingItem(
            kind="transport",
            title=f"{travel.mode.title()}: {origin} to {destination}",
            provider=provider,
            url=url,
            price_usd=travel.price_usd,
            notes=f"Departs {request.start_date}; {travel.duration_label}.",
        )

    def _hotel_items(self, plan: TravelPlan) -> list[BookingItem]:
        request = plan.request
        city = plan.destination_location.short_name if plan.destination_location else request.destination
        known = {normalise_name(h.name): h for h in plan.hotels}

        chosen: list[tuple[str, HotelOption | None]] = []
        # Hotels named in the traveller's own material take priority.
        for name in request.preferred_hotels:
            chosen.append((name, known.get(normalise_name(name))))
        if plan.recommended_hotel and all(
            normalise_name(plan.recommended_hotel.name) != normalise_name(n) for n, _ in chosen
        ):
            chosen.append((plan.recommended_hotel.name, plan.recommended_hotel))

        items = []
        for name, hotel in chosen:
            search = "https://www.booking.com/searchresults.html?" + urlencode(
                {
                    "ss": f"{name}, {city}",
                    "checkin": request.start_date.isoformat(),
                    "checkout": request.end_date.isoformat(),
                    "group_adults": request.travelers,
                    "no_rooms": 1,
                }
            )
            url = (hotel.booking_url or hotel.website) if hotel else ""
            items.append(
                BookingItem(
                    kind="accommodation",
                    title=f"{name} ({request.nights} nights)",
                    provider="Hotel website" if url else "Booking.com search",
                    url=url or search,
                    price_usd=hotel.total_price_usd if hotel else None,
                    notes="From your trip material." if name in request.preferred_hotels else "Recommended stay.",
                )
            )
        return items

    @staticmethod
    def _ticket_items(plan: TravelPlan) -> list[BookingItem]:
        scheduled = {item.place_name for day in plan.itinerary for item in day.items}
        items = []
        for place in plan.places:
            if place.name in scheduled and place.fee_required and place.website:
                items.append(
                    BookingItem(
                        kind="tickets",
                        title=f"Tickets: {place.name}",
                        provider="Official site",
                        url=place.website,
                        notes="Timed entry often sells out; book ahead.",
                    )
                )
        return items

    def success_message(self, plan: TravelPlan) -> str:
        return f"Prepared {len(plan.bookings)} booking hand-offs; no payment taken."
