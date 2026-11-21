import asyncio
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import TravelOrchestrator
from app.models.travel import TravelPlanRequest

CASES = [
    ("single day trip", "Paris", "Bruges", 0, "packed"),
    ("tiny destination", "Delhi", "Rishikesh", 2, "relaxed"),
    ("nonexistent place", "Delhi", "Zzzxqqq Nowhere Land", 2, "balanced"),
    ("long haul", "New York", "Tokyo", 5, "balanced"),
]


async def main() -> None:
    orchestrator = TravelOrchestrator()

    for label, origin, destination, extra_days, pace in CASES:
        print(f"\n{'=' * 70}\n{label}: {origin} -> {destination} (+{extra_days}d, {pace})")
        try:
            request = TravelPlanRequest(
                origin=origin,
                destination=destination,
                start_date=date.today() + timedelta(days=60),
                end_date=date.today() + timedelta(days=60 + extra_days),
                travelers=2,
                interests=["culture"],
                pace=pace,
            )
            plan = await orchestrator.plan_trip(request)
        except Exception as exc:
            print(f"  RAISED {type(exc).__name__}: {exc}")
            continue

        for result in plan.agent_results:
            print(f"  [{result.status.value:9}] {result.agent_name:18} {result.message[:70]}")
        print(f"  places={len(plan.places)} days={len(plan.itinerary)} "
              f"transport={len(plan.travel_options)} hotels={len(plan.hotels)}")
        for day in plan.itinerary:
            stops = sum(1 for i in day.items if i.activity.startswith("Visit"))
            print(f"    day {day.day}: {stops} stops, {day.walking_distance_km}km walking, {day.theme}")


asyncio.run(main())
