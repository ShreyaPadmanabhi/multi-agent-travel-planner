"""Measure what parallel dispatch actually saves, on live data.

Runs the same trip twice per round with the HTTP cache cleared in between:
once with every agent in sequence, once through the LangGraph supervisor,
which runs the four specialists concurrently. Use these numbers, not
estimates, when describing the speed-up.

    python scripts/benchmark.py "London" "Rome" --rounds 3
"""

import argparse
import asyncio
import pathlib
import statistics
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import TravelOrchestrator
from app.models.travel import TravelPlan, TravelPlanRequest
from app.providers import http


async def sequential(orchestrator: TravelOrchestrator, request: TravelPlanRequest) -> dict[str, float]:
    timings: dict[str, float] = {}
    plan = TravelPlan(request=request)
    start = time.perf_counter()
    plan, _ = await orchestrator.destination_agent.execute(plan)
    timings["discover"] = time.perf_counter() - start
    specialist_start = time.perf_counter()
    for agent in orchestrator.specialists.values():
        await agent.execute(plan.model_copy(deep=True))
    timings["specialists"] = time.perf_counter() - specialist_start
    timings["total"] = time.perf_counter() - start
    return timings


async def parallel(orchestrator: TravelOrchestrator, request: TravelPlanRequest) -> dict[str, float]:
    start = time.perf_counter()
    _, plan = await orchestrator.start(request)
    total = time.perf_counter() - start
    discover = next(r.duration_ms for r in plan.agent_results if r.agent_name == "DestinationAgent") / 1000
    branches = {r.agent_name: r.duration_ms / 1000 for r in plan.agent_results if r.agent_name != "DestinationAgent"}
    print("    branches: " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in branches.items()))
    # Branches overlap, so the specialist phase is bounded by the slowest one.
    # `branch_sum` is what the same run would have cost with the branches in sequence.
    return {
        "discover": discover,
        "specialists": total - discover,
        "branch_sum": sum(branches.values()),
        "total": total,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("origin", nargs="?", default="London")
    parser.add_argument("destination", nargs="?", default="Rome")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    start = date.today() + timedelta(days=10)
    request = TravelPlanRequest(
        origin=args.origin,
        destination=args.destination,
        start_date=start,
        end_date=start + timedelta(days=3),
        travelers=2,
        interests=["history", "food"],
    )
    orchestrator = TravelOrchestrator()

    results: dict[str, list[dict[str, float]]] = {"sequential": [], "parallel": []}
    for round_number in range(args.rounds):
        # Alternate the order so neither mode always benefits from warm upstream caches.
        modes = ["sequential", "parallel"] if round_number % 2 == 0 else ["parallel", "sequential"]
        for mode in modes:
            http._cache._store.clear()
            runner = sequential if mode == "sequential" else parallel
            timing = await runner(orchestrator, request)
            results[mode].append(timing)
            print(f"round {round_number + 1} {mode:10} " + "  ".join(f"{k}={v:6.1f}s" for k, v in timing.items()))

    print("\nMEDIAN")
    for mode, runs in results.items():
        print(f"  {mode:10} " + "  ".join(
            f"{key}={statistics.median(r[key] for r in runs):6.1f}s" for key in ("discover", "specialists", "total")
        ))
    seq = statistics.median(r["specialists"] for r in results["sequential"])
    par = statistics.median(r["specialists"] for r in results["parallel"])
    if par > 0:
        print(f"\nSpecialist phase: {seq:.1f}s sequential vs {par:.1f}s parallel ({seq / par:.1f}x)")
        # Same-run comparison: immune to upstream latency drifting between runs.
        same_run = statistics.median(r["branch_sum"] / r["specialists"] for r in results["parallel"])
        print(f"Within parallel runs, branches summed to {same_run:.1f}x the wall-clock specialist phase")


asyncio.run(main())
