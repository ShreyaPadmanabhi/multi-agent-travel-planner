"""Drive the human-in-the-loop flow end to end: draft, revise, approve."""

import asyncio
import pathlib
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.orchestrator import TravelOrchestrator
from app.models.travel import TravelPlan, TravelPlanRequest


def show(label: str, plan: TravelPlan, elapsed: float) -> None:
    print(f"\n=== {label} ({elapsed:.1f}s) status={plan.status}")
    if plan.supervisor:
        print(f"  supervisor[{plan.supervisor.mode}]: {plan.supervisor.specialists} changes={plan.supervisor.request_changes}")
        print(f"  rationale: {plan.supervisor.rationale}")
    for step in plan.supervisor_trace:
        print(f"  ReAct: {step.thought[:70]!r} -> {step.action}({step.action_input}) => {step.observation[:80]!r}")
    for result in plan.agent_results:
        print(f"  [{result.status.value:9}] {result.agent_name:16} {result.duration_ms:>6}ms {result.message}")
    print(f"  pace={plan.request.pace} must_visit={plan.request.must_visit} days={[len(d.items) for d in plan.itinerary]}")
    for booking in plan.bookings:
        print(f"  BOOK {booking.kind:13} {booking.title[:40]:42} {booking.url[:70]}")


async def main() -> None:
    orchestrator = TravelOrchestrator()
    request = TravelPlanRequest(
        origin="Florence",
        destination="Rome",
        start_date=date.today() + timedelta(days=5),
        end_date=date.today() + timedelta(days=7),
        travelers=2,
        interests=["history"],
        must_visit=["Trevi Fountain"],
        preferred_hotels=["Hotel Artemide"],
    )

    start = time.time()
    thread_id, draft = await orchestrator.start(request)
    show("DRAFT", draft, time.time() - start)
    print("  orchestrator.status:", orchestrator.status(thread_id))

    start = time.time()
    revised = await orchestrator.review(
        thread_id, "revise", feedback="Too packed, make it more relaxed and add Villa Borghese"
    )
    show("REVISED", revised, time.time() - start)

    start = time.time()
    final = await orchestrator.review(thread_id, "approve")
    show("APPROVED", final, time.time() - start)
    print("  history:", [(r.action, r.feedback) for r in final.review_history])
    print("  orchestrator.status:", orchestrator.status(thread_id))


asyncio.run(main())
