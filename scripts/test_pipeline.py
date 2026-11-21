import asyncio
import pathlib
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import TravelOrchestrator
from app.models.travel import TravelPlanRequest


async def main() -> None:
    request = TravelPlanRequest(
        origin=sys.argv[1] if len(sys.argv) > 1 else "London",
        destination=sys.argv[2] if len(sys.argv) > 2 else "Rome",
        start_date=date.today() + timedelta(days=45),
        end_date=date.today() + timedelta(days=48),
        travelers=2,
        budget_usd=2500,
        interests=["history", "food"],
        pace="balanced",
    )

    start = time.time()
    plan = await TravelOrchestrator().plan_trip(request)
    print(f"--- pipeline finished in {time.time() - start:.1f}s\n")

    print("AGENTS")
    for result in plan.agent_results:
        print(f"  [{result.status.value:9}] {result.agent_name:18} {result.duration_ms:>6}ms  {result.message}")

    print(f"\nDESTINATION: {plan.destination_label}")
    print(f"SOURCES: {plan.data_sources}")

    print(f"\nITINERARY ({len(plan.itinerary)} days)")
    for day in plan.itinerary:
        print(f"\n  Day {day.day} — {day.date} — {day.theme} (walk {day.walking_distance_km}km)")
        for item in day.items:
            travel = f"  [+{item.travel_minutes_from_previous}min {item.travel_mode}]" if item.travel_minutes_from_previous else ""
            print(f"    {item.start_time}-{item.end_time}  {item.activity[:50]}{travel}")

    print(f"\nTRANSPORT ({len(plan.travel_options)})")
    for option in plan.travel_options[:5]:
        price = f"${option.price_usd:,.2f}" if option.price_usd is not None else "no price"
        print(f"  {option.mode:7} {option.provider[:22]:24} {option.duration_label:10} {price:12} {option.source}")

    print(f"\nHOTELS ({len(plan.hotels)})")
    for hotel in plan.hotels[:5]:
        rate = f"${hotel.price_per_night_usd:,.2f}/night" if hotel.price_per_night_usd else "no rate"
        print(f"  {hotel.name[:36]:38} {str(hotel.distance_to_center_km) + 'km':9} {rate:20} {hotel.source}")

    print(f"\nCOST: {plan.cost_breakdown} -> {plan.estimated_cost_usd}")
    print(f"\nWARNINGS ({len(plan.warnings)})")
    for warning in plan.warnings:
        print(f"  - {warning}")
    print(f"\nSUMMARY\n{plan.summary}")


asyncio.run(main())
