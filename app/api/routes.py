import re
import tempfile
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.agents.orchestrator import TravelOrchestrator
from app.config import settings
from app.intake import IntakeAgent
from app.models.intake import IntakeResult
from app.models.travel import Place, TravelPlan, TravelPlanRequest
from app.providers.geocoding import GeocodingProvider
from app.providers.http import ProviderError
from app.providers.places import PlacesProvider

router = APIRouter()
orchestrator = TravelOrchestrator()
intake_agent = IntakeAgent()
geocoder = GeocodingProvider()
places_provider = PlacesProvider()

MAX_UPLOAD_BYTES = 50_000_000


class PlanTripBody(BaseModel):
    origin: str = Field(..., examples=["London"])
    destination: str = Field(..., examples=["Rome"])
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=9)
    budget_usd: float | None = Field(default=None, ge=0)
    interests: list[str] = Field(default_factory=list, examples=[["history", "food"]])
    travel_mode: str = Field(default="any", pattern="^(any|flight|ground)$")
    pace: str = Field(default="balanced", pattern="^(relaxed|balanced|packed)$")
    must_visit: list[str] = Field(default_factory=list, examples=[["Trevi Fountain"]])
    preferred_hotels: list[str] = Field(default_factory=list)
    notes: str = ""


class ReviewBody(BaseModel):
    action: str = Field(..., pattern="^(approve|revise|reject)$")
    feedback: str = Field(default="", examples=["Make it more relaxed and add Villa Borghese"])
    changes: dict = Field(default_factory=dict, examples=[{"pace": "relaxed"}])


class PlanEnvelope(BaseModel):
    thread_id: str
    status: str
    plan: TravelPlan


class IntakeResponse(BaseModel):
    intake: IntakeResult
    suggested_request: TravelPlanRequest | None = None


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "orchestration": "LangGraph supervisor with human-in-the-loop review",
        "supervisor": "ReAct over MCP tools" if orchestrator.supervisor.uses_react else "rules",
        "flight_offers": f"Duffel {settings.flight_mode}",
        "hotel_rates": False,
        "llm": settings.llm_enabled,
        "intake": {
            "captions_pdf_ocr": True,
            "whisper_and_vision": settings.llm_enabled,
        },
        "keyless_sources": ["Nominatim", "Wikidata", "Wikipedia", "OSRM", "OpenStreetMap", "Open-Meteo", "Frankfurter"],
    }


@router.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": agent.name, "description": agent.description, "sources": agent.source}
            for agent in orchestrator.agents
        ]
    }


@router.get("/places", response_model=list[Place])
async def search_places(destination: str, interests: str = "", limit: int = 20):
    """Attraction lookup on its own, without planning a full trip."""
    try:
        location = await geocoder.geocode(destination)
        return await places_provider.find_places(
            location,
            [i.strip() for i in interests.split(",") if i.strip()],
            limit=min(limit, 50),
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/plan", response_model=PlanEnvelope)
async def plan_trip(body: PlanTripBody, auto_approve: bool = False):
    """Draft a plan. It pauses for review unless `auto_approve` is set."""
    try:
        request = TravelPlanRequest(**body.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    thread_id, plan = await orchestrator.start(request)
    if plan.status == "failed":
        raise HTTPException(
            status_code=502,
            detail={"message": "No live data could be retrieved.", "warnings": plan.warnings},
        )
    if auto_approve:
        plan = await orchestrator.review(thread_id, "approve", feedback="auto-approved")
    return PlanEnvelope(thread_id=thread_id, status=orchestrator.status(thread_id) or plan.status, plan=plan)


@router.get("/plan/{thread_id}", response_model=PlanEnvelope)
async def get_plan(thread_id: str):
    plan = orchestrator.get(thread_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Unknown plan.")
    return PlanEnvelope(thread_id=thread_id, status=orchestrator.status(thread_id) or plan.status, plan=plan)


@router.post("/plan/{thread_id}/review", response_model=PlanEnvelope)
async def review_plan(thread_id: str, body: ReviewBody):
    """Approve (runs the booking agent), reject, or revise with feedback and/or field changes."""
    if orchestrator.get(thread_id) is None:
        raise HTTPException(status_code=404, detail="Unknown plan.")
    try:
        plan = await orchestrator.review(thread_id, body.action, body.feedback, body.changes)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PlanEnvelope(thread_id=thread_id, status=orchestrator.status(thread_id) or plan.status, plan=plan)


@router.post("/intake", response_model=IntakeResponse)
async def intake(
    urls: list[str] = Form(default=[]),
    text: str = Form(default=""),
    destination_hint: str = Form(default=""),
    origin: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """Extract a trip from vlogs, reels, PDFs, pamphlet photos or pasted text."""
    with tempfile.TemporaryDirectory(prefix="upload-") as tmp:
        saved: list[tuple[Path, str]] = []
        for index, upload in enumerate(files):
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{upload.filename} is larger than 50 MB.")
            # Never trust the client's filename as a path; keep only a safe suffix.
            suffix = re.sub(r"[^.\w]", "", Path(upload.filename or "").suffix.lower())[:8]
            path = Path(tmp) / f"upload-{index}{suffix}"
            path.write_bytes(data)
            saved.append((path, upload.filename or path.name))

        try:
            result = await intake_agent.process(
                urls=[u for u in urls if u.strip()], files=saved, text=text, destination_hint=destination_hint
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    suggested = None
    if result.trip.destination:
        try:
            suggested = result.to_request(origin=origin or None)
        except (ValueError, ValidationError):
            suggested = None
    return IntakeResponse(intake=result, suggested_request=suggested)
