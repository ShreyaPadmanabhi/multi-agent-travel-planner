from app.agents.base import AgentSkipped, BaseAgent
from app.models.travel import TravelOption, TravelPlan
from app.providers.flights import FlightProvider
from app.providers.http import ProviderError
from app.providers.routing import RoutingProvider

# Past this a road trip stops being a realistic alternative to flying, though
# it is still reported so the traveller can see the trade-off.
LONG_DRIVE_KM = 1500


class TransportAgent(BaseAgent):
    """Finds real ways to get from the origin to the destination."""

    name = "TransportAgent"
    description = "Searches live flight offers and computes road routes."
    source = "Duffel + OSRM"

    def __init__(
        self,
        flights: FlightProvider | None = None,
        routing: RoutingProvider | None = None,
    ) -> None:
        self.flights = flights or FlightProvider()
        self.routing = routing or RoutingProvider()

    async def run(self, plan: TravelPlan) -> TravelPlan:
        if not plan.origin_location or not plan.destination_location:
            raise AgentSkipped("Origin or destination could not be resolved.")

        request = plan.request
        origin, destination = plan.origin_location, plan.destination_location
        options: list[TravelOption] = []

        if request.travel_mode in ("any", "ground"):
            options.extend(await self._ground_options(plan))

        if request.travel_mode in ("any", "flight"):
            if self.flights.enabled:
                try:
                    options.extend(
                        await self.flights.search(
                            origin, destination, request.start_date, request.travelers
                        )
                    )
                except ProviderError as exc:
                    plan.warnings.append(f"Flight search unavailable: {exc}")
            else:
                plan.warnings.append(
                    "Flight offers need a Duffel access token (free test mode); showing road routes only."
                )

        if not options:
            raise AgentSkipped("No transport options could be retrieved for this route.")

        plan.travel_options = options
        plan.recommended_travel = self._recommend(options, request.budget_usd)
        return plan

    async def _ground_options(self, plan: TravelPlan) -> list[TravelOption]:
        origin, destination = plan.origin_location, plan.destination_location
        route = await self.routing.route(origin.coordinates, destination.coordinates)

        # A fallback result means OSRM found no road route, which is also how
        # overseas routes present themselves. Offering a drive would be wrong.
        if route.estimated:
            return []

        label = "Road route"
        if route.distance_km > LONG_DRIVE_KM:
            label = f"Road route (long drive, {route.distance_km:,.0f} km)"

        return [
            TravelOption(
                mode="drive",
                provider=label,
                origin=origin.short_name,
                destination=destination.short_name,
                duration_minutes=route.duration_minutes,
                distance_km=route.distance_km,
                stops=0,
                source=self.routing.source,
            )
        ]

    @staticmethod
    def _recommend(options: list[TravelOption], budget: float | None) -> TravelOption:
        priced = [o for o in options if o.price_usd is not None]

        if priced and budget:
            # Keep the trip's transport share sane rather than picking the
            # cheapest option regardless of how long it takes.
            affordable = [o for o in priced if o.price_usd <= budget * 0.4]
            if affordable:
                return min(affordable, key=lambda o: (o.price_usd, o.duration_minutes or 1e9))

        if priced:
            return min(priced, key=lambda o: o.price_usd)

        return min(options, key=lambda o: o.duration_minutes or 1e9)

    def success_message(self, plan: TravelPlan) -> str:
        priced = sum(1 for o in plan.travel_options if o.price_usd is not None)
        return f"Found {len(plan.travel_options)} transport options ({priced} with live prices)."
