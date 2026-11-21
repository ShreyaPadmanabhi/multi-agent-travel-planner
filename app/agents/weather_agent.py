from app.agents.base import AgentSkipped, BaseAgent
from app.models.travel import TravelPlan
from app.providers.weather import WeatherProvider

WET_DAY_MM = 5.0


class WeatherAgent(BaseAgent):
    """Looks up daily weather for the trip dates at the destination."""

    name = "WeatherAgent"
    description = "Daily forecast, or last year's weather when the trip is beyond the forecast window."
    source = "Open-Meteo"

    def __init__(self, weather: WeatherProvider | None = None) -> None:
        self.weather = weather or WeatherProvider()

    async def run(self, plan: TravelPlan) -> TravelPlan:
        if not plan.destination_location:
            raise AgentSkipped("Destination could not be resolved.")

        request = plan.request
        plan.weather = await self.weather.daily(
            plan.destination_location, request.start_date, request.end_date
        )

        if plan.weather and plan.weather[0].basis != "forecast":
            plan.warnings.append(
                "Trip dates are beyond the forecast window; weather shows the same dates last year."
            )
        wet = [d for d in plan.weather if (d.precipitation_mm or 0) >= WET_DAY_MM]
        if wet:
            days = ", ".join(f"{d.date:%a %d %b}" for d in wet)
            plan.warnings.append(f"Wet weather likely on {days}; consider indoor stops those days.")
        return plan

    def success_message(self, plan: TravelPlan) -> str:
        basis = plan.weather[0].basis if plan.weather else "none"
        return f"Weather for {len(plan.weather)} days ({basis})."
