from app.agents.base import AgentSkipped, BaseAgent
from app.models.travel import HotelOption, TravelPlan
from app.providers.hotels import HotelProvider


class HotelAgent(BaseAgent):
    """Finds real accommodation near the destination."""

    name = "HotelAgent"
    description = "Finds real, mapped accommodation near the centre."
    source = "OpenStreetMap"

    def __init__(self, hotels: HotelProvider | None = None) -> None:
        self.hotels = hotels or HotelProvider()

    async def run(self, plan: TravelPlan) -> TravelPlan:
        if not plan.destination_location:
            raise AgentSkipped("Destination could not be resolved.")

        request = plan.request
        hotels = await self.hotels.search(
            plan.destination_location,
            request.start_date,
            request.end_date,
            request.travelers,
        )

        if not hotels:
            raise AgentSkipped("No mapped accommodation was found near the destination.")

        if not any(h.total_price_usd is not None for h in hotels):
            plan.warnings.append(
                "Showing real properties without nightly rates; the booking links open a live price search."
            )

        plan.hotels = hotels
        plan.recommended_hotel = self._recommend(hotels, request.budget_usd, request.nights)
        return plan

    @staticmethod
    def _recommend(
        hotels: list[HotelOption], budget: float | None, nights: int
    ) -> HotelOption:
        priced = [h for h in hotels if h.total_price_usd is not None]

        if priced and budget:
            # Lodging is capped at 35% of the trip budget so that transport and
            # daily spending still fit.
            ceiling = budget * 0.35
            affordable = [h for h in priced if h.total_price_usd <= ceiling]
            if affordable:
                return min(
                    affordable,
                    key=lambda h: (h.distance_to_center_km or 1e9, h.total_price_usd),
                )

        if priced:
            return min(priced, key=lambda h: h.total_price_usd)

        return min(hotels, key=lambda h: h.distance_to_center_km or 1e9)

    def success_message(self, plan: TravelPlan) -> str:
        priced = sum(1 for h in plan.hotels if h.total_price_usd is not None)
        return f"Found {len(plan.hotels)} places to stay ({priced} with live rates)."
