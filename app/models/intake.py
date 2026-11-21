"""Shared schema produced by the multimodal intake agent.

Whatever the source — a YouTube vlog, a reel, a PDF itinerary or a photographed
pamphlet — intake ends in a `TripIntake`. `IntakeResult.to_request` turns it
into the `TravelPlanRequest` the planning graph consumes: named attractions
become pinned must-visit stops for the itinerary agent, and named hotels become
preferred stays for the booking agent.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.travel import TravelPlanRequest

DEFAULT_TRIP_DAYS = 3


class IntakePlace(BaseModel):
    name: str
    kind: Literal["attraction", "food", "hotel", "area", "activity"] = "attraction"
    day: Optional[int] = Field(default=None, description="Day number if the source assigns one")
    notes: str = ""


class TripIntake(BaseModel):
    destination: Optional[str] = None
    origin: Optional[str] = None
    start_date: Optional[date] = None
    duration_days: Optional[int] = Field(default=None, ge=1, le=30)
    travelers: Optional[int] = Field(default=None, ge=1, le=9)
    budget_usd: Optional[float] = Field(default=None, ge=0)
    interests: list[str] = Field(default_factory=list)
    pace: Optional[Literal["relaxed", "balanced", "packed"]] = None
    places: list[IntakePlace] = Field(default_factory=list)
    transport_notes: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class SourceReport(BaseModel):
    """What was pulled out of one input, and how."""

    kind: str = Field(description="youtube, video, pdf, image, or text")
    reference: str
    methods: list[str] = Field(default_factory=list, description="e.g. captions, whisper, ocr, vision")
    characters: int = 0
    warnings: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    trip: TripIntake
    sources: list[SourceReport] = Field(default_factory=list)
    extraction: Literal["llm", "heuristic"] = "heuristic"
    warnings: list[str] = Field(default_factory=list)
    evidence_excerpt: str = ""

    def to_request(
        self,
        origin: str | None = None,
        start_date: date | None = None,
        travelers: int | None = None,
        destination: str | None = None,
    ) -> TravelPlanRequest:
        """Build a planning request; explicit arguments override what intake found."""
        trip = self.trip
        destination = destination or trip.destination
        if not destination:
            raise ValueError("The trip material did not name a destination; please provide one.")

        start = start_date or trip.start_date or date.today() + timedelta(days=30)
        days = trip.duration_days or DEFAULT_TRIP_DAYS
        stops = [p.name for p in trip.places if p.kind != "hotel"]
        hotels = [p.name for p in trip.places if p.kind == "hotel"]

        return TravelPlanRequest(
            origin=origin or trip.origin or "",
            destination=destination,
            start_date=start,
            end_date=start + timedelta(days=days - 1),
            travelers=travelers or trip.travelers or 1,
            budget_usd=trip.budget_usd,
            interests=trip.interests,
            pace=trip.pace or "balanced",
            must_visit=list(dict.fromkeys(stops)),
            preferred_hotels=list(dict.fromkeys(hotels)),
            notes=" ".join(trip.tips + trip.transport_notes)[:1000],
        )
