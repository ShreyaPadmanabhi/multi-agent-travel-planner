"""LangGraph workflow for the travel-planning agents.

A supervisor routes the work. Discovery runs first because every specialist
needs the resolved destination; the supervisor then fans out to the four
specialists, which LangGraph runs concurrently in one superstep:

    supervisor -> discover -> supervisor -> itinerary  \\
                                          -> transport  \\
                                          -> hotel       -> finalize -> human_review
                                          -> weather    /                 |
                                                                approve -> booking -> END
                                                                revise  -> supervisor
                                                                reject  -> END

`human_review` calls `interrupt()`, so the graph pauses with its state in the
checkpointer until the traveller approves, rejects, or asks for changes. On a
revision the supervisor re-runs only the specialists the change affects and
keeps the other branches' results from the checkpoint.

Each parallel node works on a deep copy and returns a distinct state key. This
avoids concurrent mutation of the same Pydantic model.
"""

from __future__ import annotations

import uuid
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents.base import BaseAgent
from app.agents.booking_agent import BookingAgent
from app.agents.destination_agent import DestinationAgent
from app.agents.hotel_agent import HotelAgent
from app.agents.itinerary_agent import ItineraryAgent
from app.agents.supervisor import SPECIALISTS, Supervisor, apply_changes, impact_of
from app.agents.transport_agent import TransportAgent
from app.agents.weather_agent import WeatherAgent
from app.models import travel as travel_models
from app.models.travel import (
    AgentResult,
    AgentStatus,
    ReasoningStep,
    ReviewRecord,
    SupervisorDecision,
    TravelPlan,
    TravelPlanRequest,
)


def _checkpoint_serializer() -> JsonPlusSerializer:
    """Allow the plan models, and nothing else of ours, to be restored from checkpoints."""
    names = [
        name
        for name, value in vars(travel_models).items()
        if isinstance(value, type) and value.__module__ == travel_models.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=[(travel_models.__name__, n) for n in names])


class BranchOutput(TypedDict):
    plan: TravelPlan
    result: AgentResult
    warnings: list[str]


class TravelGraphState(TypedDict, total=False):
    request: TravelPlanRequest
    plan: TravelPlan
    draft: TravelPlan
    needs_discovery: bool
    pending_review: ReviewRecord | None
    pending_decision: SupervisorDecision | None
    decision: SupervisorDecision
    trace: list[ReasoningStep]
    review_history: list[ReviewRecord]
    itinerary_output: BranchOutput
    transport_output: BranchOutput
    hotel_output: BranchOutput
    weather_output: BranchOutput


class TravelOrchestrator:
    """Builds the compiled LangGraph workflow and drives its review loop."""

    def __init__(
        self,
        destination_agent: DestinationAgent | None = None,
        itinerary_agent: ItineraryAgent | None = None,
        transport_agent: TransportAgent | None = None,
        hotel_agent: HotelAgent | None = None,
        weather_agent: WeatherAgent | None = None,
        booking_agent: BookingAgent | None = None,
        supervisor: Supervisor | None = None,
        checkpointer=None,
    ) -> None:
        self.destination_agent = destination_agent or DestinationAgent()
        self.specialists: dict[str, BaseAgent] = {
            "itinerary": itinerary_agent or ItineraryAgent(),
            "transport": transport_agent or TransportAgent(),
            "hotel": hotel_agent or HotelAgent(),
            "weather": weather_agent or WeatherAgent(),
        }
        self.booking_agent = booking_agent or BookingAgent()
        self.supervisor = supervisor or Supervisor()
        # In-memory checkpoints live as long as the process. Swap in a SQLite or
        # Postgres saver to keep paused plans across restarts.
        self.checkpointer = checkpointer or InMemorySaver(serde=_checkpoint_serializer())
        self.graph = self._build_graph()

    @property
    def agents(self) -> list[BaseAgent]:
        return [self.destination_agent, *self.specialists.values(), self.booking_agent]

    # ------------------------------------------------------------------ API

    async def start(self, request: TravelPlanRequest, thread_id: str | None = None) -> tuple[str, TravelPlan]:
        """Run until the draft is ready for review. Returns the thread id and draft."""
        thread_id = thread_id or uuid.uuid4().hex
        state = await self.graph.ainvoke(
            {"request": request, "needs_discovery": True, "trace": [], "review_history": []},
            self._config(thread_id),
        )
        return thread_id, state["draft"]

    async def review(
        self, thread_id: str, action: str, feedback: str = "", changes: dict | None = None
    ) -> TravelPlan:
        """Resume a paused plan with the traveller's decision."""
        if action not in ("approve", "revise", "reject"):
            raise ValueError("action must be approve, revise, or reject")
        if self.status(thread_id) != "awaiting_approval":
            raise LookupError(f"Plan {thread_id} is not awaiting approval.")
        state = await self.graph.ainvoke(
            Command(resume={"action": action, "feedback": feedback, "changes": changes or {}}),
            self._config(thread_id),
        )
        return state["draft"]

    def get(self, thread_id: str) -> TravelPlan | None:
        snapshot = self.graph.get_state(self._config(thread_id))
        return snapshot.values.get("draft") if snapshot and snapshot.values else None

    def status(self, thread_id: str) -> str | None:
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot or not snapshot.values:
            return None
        if snapshot.next and "human_review" in snapshot.next:
            return "awaiting_approval"
        draft = snapshot.values.get("draft")
        return draft.status if draft else None

    async def plan_trip(self, request: TravelPlanRequest, auto_approve: bool = True) -> TravelPlan:
        """One-shot planning for scripts: start, then approve without a human."""
        thread_id, draft = await self.start(request)
        if auto_approve and self.status(thread_id) == "awaiting_approval":
            return await self.review(thread_id, "approve", feedback="auto-approved")
        return draft

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    # ---------------------------------------------------------------- graph

    def _build_graph(self):
        workflow = StateGraph(TravelGraphState)
        workflow.add_node("supervisor", self._supervise)
        workflow.add_node("discover", self._discover)
        for name in SPECIALISTS:
            workflow.add_node(name, self._specialist_node(name))
            workflow.add_edge(name, "finalize")
        workflow.add_node("finalize", self._finalize)
        workflow.add_node("human_review", self._human_review)
        workflow.add_node("booking", self._book)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("booking", END)
        return workflow.compile(checkpointer=self.checkpointer)

    async def _supervise(
        self, state: TravelGraphState
    ) -> Command[Literal["discover", "itinerary", "transport", "hotel", "weather", "finalize"]]:
        if state.get("needs_discovery"):
            return Command(goto="discover")

        trace = list(state.get("trace", []))
        decision = state.get("pending_decision")
        if decision is None:
            # A fresh decision: the first pass, or a revision the traveller asked for.
            review = state.get("pending_review")
            decision, steps = await self.supervisor.decide(state["plan"], review)
            trace.extend(steps)

            if decision.request_changes:
                request, _ = apply_changes(state["request"], decision.request_changes)
                _, rediscover = impact_of(decision.request_changes)
                update = {"request": request, "pending_review": None, "trace": trace}
                if rediscover:
                    # Discover with the new request, then come back to dispatch this decision.
                    return Command(
                        goto="discover",
                        update={**update, "pending_decision": decision, "needs_discovery": True},
                    )
                plan = state["plan"].model_copy(deep=True)
                plan.request = request
                update["plan"] = plan
            else:
                update = {"pending_review": None, "trace": trace}
        else:
            update = {"trace": trace}

        update.update({"decision": decision, "pending_decision": None})
        return Command(goto=decision.specialists or "finalize", update=update)

    async def _discover(self, state: TravelGraphState) -> Command[Literal["supervisor", "finalize"]]:
        plan = TravelPlan(request=state["request"])
        plan, result = await self.destination_agent.execute(plan)
        update = {"plan": plan, "needs_discovery": False}
        if result.status is AgentStatus.FAILED:
            return Command(goto="finalize", update=update)
        return Command(goto="supervisor", update=update)

    def _specialist_node(self, name: str):
        agent = self.specialists[name]

        async def run(state: TravelGraphState) -> TravelGraphState:
            working_plan = state["plan"].model_copy(deep=True)
            warning_count = len(working_plan.warnings)
            working_plan, result = await agent.execute(working_plan)
            output: BranchOutput = {
                "plan": working_plan,
                "result": result,
                "warnings": working_plan.warnings[warning_count:],
            }
            return {f"{name}_output": output}

        run.__name__ = f"run_{name}"
        return run

    async def _finalize(self, state: TravelGraphState) -> Command[Literal["human_review", "__end__"]]:
        plan = state["plan"].model_copy(deep=True)
        discovered = plan.destination_location is not None

        outputs = {name: state.get(f"{name}_output") for name in SPECIALISTS} if discovered else {}
        if itinerary := outputs.get("itinerary"):
            plan.itinerary = itinerary["plan"].itinerary
        if transport := outputs.get("transport"):
            plan.travel_options = transport["plan"].travel_options
            plan.recommended_travel = transport["plan"].recommended_travel
        if hotel := outputs.get("hotel"):
            plan.hotels = hotel["plan"].hotels
            plan.recommended_hotel = hotel["plan"].recommended_hotel
        if weather := outputs.get("weather"):
            plan.weather = weather["plan"].weather
        for output in outputs.values():
            if output:
                plan.agent_results.append(output["result"])
                plan.warnings.extend(output["warnings"])

        decision = state.get("decision")
        plan.supervisor = decision
        if decision:
            plan.warnings.extend(decision.advisories)
        plan.supervisor_trace = list(state.get("trace", []))
        plan.review_history = list(state.get("review_history", []))

        if any(outputs.values()):
            self._estimate_cost(plan)
        plan.data_sources = self._collect_sources(plan)
        plan.summary = self._summarise(plan)

        if not discovered:
            plan.status = "failed"
            return Command(goto=END, update={"draft": plan})
        plan.status = "awaiting_approval"
        return Command(goto="human_review", update={"draft": plan})

    async def _human_review(self, state: TravelGraphState) -> Command[Literal["supervisor", "booking", "__end__"]]:
        draft = state["draft"]
        answer = interrupt(
            {
                "question": "Approve this plan, ask for changes, or reject it?",
                "summary": draft.summary,
                "editable_fields": list(TravelPlanRequest.model_fields),
            }
        )
        review = ReviewRecord(**answer)
        history = list(state.get("review_history", [])) + [review]

        if review.action == "approve":
            return Command(goto="booking", update={"review_history": history})
        if review.action == "reject":
            rejected = draft.model_copy(deep=True)
            rejected.status = "rejected"
            rejected.review_history = history
            return Command(goto=END, update={"draft": rejected, "review_history": history})
        return Command(goto="supervisor", update={"review_history": history, "pending_review": review})

    async def _book(self, state: TravelGraphState) -> TravelGraphState:
        plan = state["draft"].model_copy(deep=True)
        plan.status = "approved"
        plan.review_history = list(state.get("review_history", []))
        plan, _ = await self.booking_agent.execute(plan)
        plan.summary = self._summarise(plan)
        return {"draft": plan}

    # -------------------------------------------------------------- helpers

    def _estimate_cost(self, plan: TravelPlan) -> None:
        """Total only what has a real price attached; never invent a figure."""
        breakdown: dict[str, float] = {}

        travel = plan.recommended_travel
        if travel and travel.price_usd is not None:
            breakdown["transport"] = round(travel.price_usd, 2)

        hotel = plan.recommended_hotel
        if hotel and hotel.total_price_usd is not None:
            breakdown["accommodation"] = round(hotel.total_price_usd, 2)

        plan.cost_breakdown = breakdown
        plan.estimated_cost_usd = round(sum(breakdown.values()), 2) if breakdown else None

        if plan.estimated_cost_usd is None:
            plan.warnings.append(
                "No live prices were available, so the trip cost could not be estimated."
            )
        elif "transport" not in breakdown or "accommodation" not in breakdown:
            plan.warnings.append(
                "Cost estimate is partial — it covers only the components with live pricing."
            )

    @staticmethod
    def _collect_sources(plan: TravelPlan) -> list[str]:
        sources: list[str] = []
        for value in (
            [p.source for p in plan.places]
            + [o.source for o in plan.travel_options]
            + [h.source for h in plan.hotels]
            + (["Open-Meteo"] if plan.weather else [])
        ):
            if value and value not in sources:
                sources.append(value)
        return sources

    def _summarise(self, plan: TravelPlan) -> str:
        request = plan.request
        destination = plan.destination_label.split(",")[0] if plan.destination_label else request.destination

        lines = [
            f"{request.days}-day trip to {destination} from {request.origin}",
            f"{request.start_date} to {request.end_date} for {request.travelers} traveller(s)",
            f"{len(plan.places)} places found, {len(plan.itinerary)} days scheduled",
        ]

        if plan.recommended_travel:
            travel = plan.recommended_travel
            price = f"${travel.price_usd:,.0f}" if travel.price_usd is not None else "price unavailable"
            lines.append(f"Transport: {travel.provider} ({travel.mode}), {travel.duration_label}, {price}")

        if plan.recommended_hotel:
            hotel = plan.recommended_hotel
            if hotel.total_price_usd is not None:
                lines.append(f"Stay: {hotel.name} — ${hotel.total_price_usd:,.0f} total")
            else:
                lines.append(f"Stay: {hotel.name} (no live rate)")

        if plan.weather:
            highs = [d.temp_max_c for d in plan.weather if d.temp_max_c is not None]
            if highs:
                lines.append(f"Weather: highs {min(highs):.0f}–{max(highs):.0f}°C ({plan.weather[0].basis})")

        if plan.estimated_cost_usd is not None:
            lines.append(f"Priced components total ${plan.estimated_cost_usd:,.2f}")

        if plan.bookings:
            lines.append(f"{len(plan.bookings)} booking hand-offs ready")

        return "\n".join(lines)
