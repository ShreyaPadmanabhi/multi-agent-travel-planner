"""Supervisor that decides which specialists run, and with what request.

With an OpenAI key the supervisor runs a ReAct loop: it writes a thought,
calls live-data tools over MCP, reads the observation, and repeats until it
calls `dispatch_specialists`. The model proposes; deterministic rules dispose.
Whatever the model says, every specialist whose inputs changed is re-run, so a
bad or missing LLM answer can never leave a stale section in the plan.

Without a key, or if the loop fails, the same rules make the decision alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

from pydantic import ValidationError

from app.config import settings
from app.mcp import MCPToolbox
from app.models.travel import (
    ReasoningStep,
    ReviewRecord,
    SupervisorDecision,
    TravelPlan,
    TravelPlanRequest,
)

logger = logging.getLogger(__name__)

SPECIALISTS = ("itinerary", "transport", "hotel", "weather")

# Which specialists a changed request field invalidates, and whether the
# destination agent must re-run first.
FIELD_IMPACT: dict[str, tuple[set[str], bool]] = {
    "destination": (set(SPECIALISTS), True),
    "start_date": (set(SPECIALISTS), True),
    "end_date": (set(SPECIALISTS), True),
    "origin": ({"transport"}, True),
    "interests": ({"itinerary"}, True),
    "must_visit": ({"itinerary"}, True),
    "pace": ({"itinerary"}, False),
    "travel_mode": ({"transport"}, False),
    "budget_usd": ({"transport", "hotel"}, False),
    "travelers": ({"transport", "hotel"}, False),
    "preferred_hotels": ({"hotel"}, False),
}
EDITABLE_FIELDS = tuple(FIELD_IMPACT)

SYSTEM_PROMPT = """You are the supervisor of a travel-planning team. Your specialists are:
- itinerary: schedules attractions into days (needs the discovered places)
- transport: flights and road routes from origin to destination
- hotel: accommodation near the centre
- weather: daily weather for the trip dates

Work in ReAct style. Before every tool call, write one short line starting with "Thought:".
Use the live-data tools to check what matters for this trip (for example whether a road route
exists, whether flights are available, what the weather looks like) — at most 4 tool calls.
Then call dispatch_specialists exactly once.

When the traveller has given review feedback, translate it into request_changes using only these
fields: origin, destination, start_date, end_date (YYYY-MM-DD), travelers, budget_usd, interests,
travel_mode (any|flight|ground), pace (relaxed|balanced|packed), must_visit, preferred_hotels.
List fields replace the old value, so include existing entries you want to keep.
Never change the request on the first pass; put concerns in advisories instead.
Advisories are short, concrete warnings for the traveller, grounded in tool observations."""

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch_specialists",
        "description": "Finish reasoning: choose which specialists run and any request changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "specialists": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(SPECIALISTS)},
                },
                "request_changes": {"type": "object"},
                "rationale": {"type": "string"},
                "advisories": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["specialists", "rationale"],
        },
    },
}


def impact_of(changes: dict) -> tuple[set[str], bool]:
    specialists: set[str] = set()
    needs_discovery = False
    for field in changes:
        affected, rediscover = FIELD_IMPACT.get(field, (set(), False))
        specialists |= affected
        needs_discovery = needs_discovery or rediscover
    return specialists, needs_discovery


def apply_changes(request: TravelPlanRequest, changes: dict) -> tuple[TravelPlanRequest, dict]:
    """Return the updated request and the subset of changes that validated and differ."""
    current = request.model_dump()
    accepted: dict = {}
    for field, value in changes.items():
        if field not in EDITABLE_FIELDS:
            continue
        candidate = {**current, **accepted, field: value}
        try:
            validated = TravelPlanRequest(**candidate)
        except ValidationError:
            logger.info("Rejected change %s=%r", field, value)
            continue
        if getattr(validated, field) != getattr(request, field):
            accepted[field] = getattr(validated, field)
    if not accepted:
        return request, {}
    return TravelPlanRequest(**{**current, **accepted}), accepted


def parse_feedback_rules(feedback: str, request: TravelPlanRequest) -> dict:
    """Keyword interpretation of free-text feedback, used when no LLM is configured."""
    text = feedback.lower()
    changes: dict[str, Any] = {}

    if re.search(r"\b(relax|slower|fewer|less rushed|too (much|busy|packed))", text):
        changes["pace"] = "relaxed"
    elif re.search(r"\b(packed|more stops|busier|see more)\b", text):
        changes["pace"] = "packed"

    if re.search(r"\b(cheaper|lower budget|too expensive|save money)\b", text) and request.budget_usd:
        changes["budget_usd"] = round(request.budget_usd * 0.8, 2)

    if re.search(r"\b(fly|flight|plane)\b", text):
        changes["travel_mode"] = "flight"
    elif re.search(r"\b(drive|driving|train|road|ground)\b", text):
        changes["travel_mode"] = "ground"

    added = re.findall(r"\b(?:add|include|visit|see)\s+(?:the\s+)?([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*)", feedback)
    if added:
        changes["must_visit"] = list(dict.fromkeys(request.must_visit + added))
    return changes


class Supervisor:
    def __init__(
        self,
        client: Any | None = None,
        toolbox_factory: Callable[[], MCPToolbox] = MCPToolbox,
    ) -> None:
        self._client = client
        self._toolbox_factory = toolbox_factory
        if self._client is None and settings.llm_enabled:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            except Exception as exc:
                logger.warning("OpenAI client unavailable: %s", exc)

    @property
    def uses_react(self) -> bool:
        return self._client is not None

    async def decide(
        self, plan: TravelPlan, review: ReviewRecord | None = None
    ) -> tuple[SupervisorDecision, list[ReasoningStep]]:
        """Choose specialists and request changes for the first pass or a revision."""
        explicit = dict(review.changes) if review else {}

        proposal: SupervisorDecision | None = None
        trace: list[ReasoningStep] = []
        advisories: list[str] = []
        if self._client is not None:
            try:
                proposal, trace = await asyncio.wait_for(self._react(plan, review), timeout=120)
            except Exception as exc:
                logger.warning("ReAct supervisor failed, using rules: %s", exc)
                advisories.append(f"Supervisor fell back to rules: {type(exc).__name__}.")

        if review is None:
            proposed_changes: dict = {}
        elif proposal is not None:
            proposed_changes = proposal.request_changes
        else:
            proposed_changes = parse_feedback_rules(review.feedback, plan.request)

        # Explicit edits from the review form win over interpreted feedback.
        _, accepted = apply_changes(plan.request, {**proposed_changes, **explicit})
        required, _ = impact_of(accepted)

        if review is None:
            specialists = set(SPECIALISTS)
        else:
            chosen = set(proposal.specialists) if proposal else set()
            specialists = (chosen & set(SPECIALISTS)) | required

        if review is not None and not specialists:
            advisories.append("No actionable change was found in the feedback; the plan is unchanged.")

        decision = SupervisorDecision(
            specialists=[s for s in SPECIALISTS if s in specialists],
            request_changes={k: _jsonable(v) for k, v in accepted.items()},
            rationale=(proposal.rationale if proposal else self._rules_rationale(review, accepted)),
            advisories=(proposal.advisories if proposal else []) + advisories,
            mode="react" if proposal else "rules",
        )
        return decision, trace

    @staticmethod
    def _rules_rationale(review: ReviewRecord | None, accepted: dict) -> str:
        if review is None:
            return "First pass: every specialist runs in parallel on the discovered destination."
        if not accepted:
            return "Revision requested, but nothing in the request changed."
        return "Re-running only the specialists affected by: " + ", ".join(accepted) + "."

    async def _react(
        self, plan: TravelPlan, review: ReviewRecord | None
    ) -> tuple[SupervisorDecision | None, list[ReasoningStep]]:
        trace: list[ReasoningStep] = []
        context = {
            "phase": "revision" if review else "first_pass",
            "request": plan.request.model_dump(mode="json"),
            "destination_resolved": plan.destination_label,
            "places_found": len(plan.places),
            "requested_places_pinned": sum(p.pinned for p in plan.places),
            "capabilities": {
                "flight_offers": settings.flight_mode,
                "hotel_rates": False,
            },
        }
        if review:
            context["review_feedback"] = review.feedback
            context["review_form_changes"] = {k: _jsonable(v) for k, v in review.changes.items()}

        async with self._toolbox_factory() as toolbox:
            tools = toolbox.openai_tools() + [DISPATCH_TOOL]
            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, default=str)},
            ]

            for _ in range(settings.supervisor_max_steps):
                response = await self._client.chat.completions.create(
                    model=settings.openai_supervisor_model,
                    messages=messages,
                    tools=tools,
                    temperature=0.2,
                )
                message = response.choices[0].message
                thought = _strip_thought(message.content or "")
                calls = message.tool_calls or []
                if not calls:
                    trace.append(ReasoningStep(thought=thought, action="final_answer_without_dispatch"))
                    return None, trace

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments},
                            }
                            for c in calls
                        ],
                    }
                )

                for call in calls:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}

                    if call.function.name == "dispatch_specialists":
                        trace.append(
                            ReasoningStep(thought=thought, action="dispatch_specialists", action_input=arguments)
                        )
                        return (
                            SupervisorDecision(
                                specialists=[s for s in arguments.get("specialists", []) if s in SPECIALISTS],
                                request_changes=arguments.get("request_changes") or {},
                                rationale=str(arguments.get("rationale", ""))[:500],
                                advisories=[str(a)[:240] for a in arguments.get("advisories", [])][:5],
                                mode="react",
                            ),
                            trace,
                        )

                    try:
                        observation = await toolbox.call(call.function.name, arguments)
                    except Exception as exc:
                        observation = f"Error: {type(exc).__name__}: {exc}"
                    trace.append(
                        ReasoningStep(
                            thought=thought,
                            action=call.function.name,
                            action_input=arguments,
                            observation=observation[:400],
                        )
                    )
                    thought = ""
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

        trace.append(ReasoningStep(thought="Step limit reached.", action="stop"))
        return None, trace


def _strip_thought(text: str) -> str:
    return re.sub(r"^\s*thought:\s*", "", text.strip(), flags=re.IGNORECASE)[:400]


def _jsonable(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value
