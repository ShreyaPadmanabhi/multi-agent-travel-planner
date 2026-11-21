"""Exercise the supervisor's ReAct loop against the real MCP server.

The LLM is replaced by a scripted fake so this runs without an OpenAI key: it
emits a thought plus a tool call, reads the observation, then dispatches. The
MCP subprocess, tool schemas and observations are all real.
"""

import asyncio
import json
import pathlib
import sys
from datetime import date, timedelta
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.supervisor import Supervisor
from app.models.travel import ReviewRecord, TravelPlan, TravelPlanRequest


def _message(content, calls):
    tool_calls = [
        SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.seen_tools: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.seen_tools = [t["function"]["name"] for t in kwargs["tools"]]
        observations = [m for m in kwargs["messages"] if m["role"] == "tool"]
        if observations:
            print("  observation fed back:", observations[-1]["content"][:120])
        return self.script.pop(0)


async def main() -> None:
    start = date.today() + timedelta(days=4)
    request = TravelPlanRequest(
        origin="London", destination="Rome", start_date=start, end_date=start + timedelta(days=2), pace="balanced"
    )
    plan = TravelPlan(request=request, destination_label="Rome, Italy")

    llm = ScriptedLLM(
        [
            _message("Thought: London to Rome crosses the Channel; check for a road route.",
                     [("find_route", {"origin": "London", "destination": "Rome"})]),
            _message("Thought: The traveller wants fewer stops, so only the itinerary changes.",
                     [("dispatch_specialists", {
                         "specialists": ["itinerary"],
                         "request_changes": {"pace": "relaxed", "budget_usd": "not a number"},
                         "rationale": "Pace change affects only the schedule.",
                         "advisories": ["A long drive is possible, but flying is far quicker."],
                     })]),
        ]
    )
    supervisor = Supervisor(client=llm)
    review = ReviewRecord(action="revise", feedback="It's too rushed, slow it down.")
    decision, trace = await supervisor.decide(plan, review)

    print("tools offered to the model:", llm.seen_tools)
    for step in trace:
        print(f"  {step.thought!r} -> {step.action}({step.action_input}) => {step.observation[:80]!r}")
    print("decision:", decision.model_dump())
    assert decision.mode == "react"
    assert decision.request_changes == {"pace": "relaxed"}, "invalid budget must be rejected"
    assert decision.specialists == ["itinerary"]
    print("OK")


asyncio.run(main())
