from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TravelPlanRequest(BaseModel):
    origin: str = Field(..., description="Departure city")
    destination: str = Field(..., description="Target destination")
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=9)
    budget_usd: Optional[float] = Field(default=None, ge=0)
    interests: list[str] = Field(default_factory=list)
    travel_mode: str = Field(default="any", description="any, flight, or ground")
    pace: str = Field(default="balanced", description="relaxed, balanced, or packed")
    must_visit: list[str] = Field(
        default_factory=list, description="Places the traveller insists on, e.g. from a vlog or pamphlet"
    )
    preferred_hotels: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def check_dates(self) -> "TravelPlanRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self

    @property
    def nights(self) -> int:
        return max((self.end_date - self.start_date).days, 1)

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1


class GeoLocation(BaseModel):
    """A resolved point on the map, as returned by the geocoder."""

    name: str
    display_name: str
    latitude: float
    longitude: float
    country_code: str = ""

    @property
    def short_name(self) -> str:
        return self.display_name.split(",")[0].strip() or self.name

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.latitude, self.longitude


class Place(BaseModel):
    name: str
    category: str
    description: str = ""
    estimated_hours: float = 1.5
    address: str = ""
    latitude: float
    longitude: float
    wikipedia_title: str = ""
    website: str = ""
    opening_hours: str = ""
    fee_required: bool = False
    image_url: str = ""
    popularity: int = Field(default=0, description="Wikidata sitelink count, used as a notability proxy")
    pinned: bool = Field(default=False, description="Requested by the traveller; never dropped from the schedule")
    source: str = ""


class ItineraryItem(BaseModel):
    start_time: str
    end_time: str
    activity: str
    place_name: str
    notes: str = ""
    travel_minutes_from_previous: int = 0
    travel_mode: str = ""


class DayItinerary(BaseModel):
    day: int
    date: date
    theme: str
    items: list[ItineraryItem] = Field(default_factory=list)
    walking_distance_km: float = 0.0


class FlightSegment(BaseModel):
    carrier: str
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime


class TravelOption(BaseModel):
    mode: str = Field(description="flight, drive, or transit")
    provider: str
    origin: str
    destination: str
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    distance_km: Optional[float] = None
    price_usd: Optional[float] = Field(default=None, description="None when no live price is available")
    stops: Optional[int] = None
    booking_class: str = ""
    segments: list[FlightSegment] = Field(default_factory=list)
    booking_url: str = ""
    source: str = ""

    @property
    def duration_label(self) -> str:
        if self.duration_minutes is None:
            return "unknown"
        hours, minutes = divmod(self.duration_minutes, 60)
        return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


class HotelOption(BaseModel):
    name: str
    stars: Optional[int] = Field(default=None, ge=1, le=5)
    address: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    price_per_night_usd: Optional[float] = None
    total_price_usd: Optional[float] = None
    currency: str = "USD"
    room_description: str = ""
    amenities: list[str] = Field(default_factory=list)
    distance_to_center_km: Optional[float] = None
    website: str = ""
    booking_url: str = ""
    source: str = ""


class DayWeather(BaseModel):
    date: date
    summary: str
    temp_max_c: Optional[float] = None
    temp_min_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    precipitation_probability: Optional[int] = None
    basis: str = Field(description="forecast, or same-dates-last-year when beyond the forecast window")


class BookingItem(BaseModel):
    """A booking hand-off: a prefilled provider link. No payment is ever taken."""

    kind: str = Field(description="transport, accommodation, or tickets")
    title: str
    provider: str
    url: str
    price_usd: Optional[float] = None
    status: str = "ready"
    notes: str = ""


class ReasoningStep(BaseModel):
    """One Thought -> Action -> Observation turn of the supervisor's ReAct loop."""

    thought: str = ""
    action: str
    action_input: dict = Field(default_factory=dict)
    observation: str = ""


class SupervisorDecision(BaseModel):
    specialists: list[str]
    request_changes: dict = Field(default_factory=dict)
    rationale: str = ""
    advisories: list[str] = Field(default_factory=list)
    mode: str = Field(default="rules", description="react when an LLM drove the decision, otherwise rules")


class ReviewRecord(BaseModel):
    action: str = Field(description="approve, revise, or reject")
    feedback: str = ""
    changes: dict = Field(default_factory=dict)
    reviewed_at: datetime = Field(default_factory=datetime.now)


class AgentResult(BaseModel):
    agent_name: str
    status: AgentStatus
    message: str
    duration_ms: int = 0
    source: str = ""


class TravelPlan(BaseModel):
    request: TravelPlanRequest
    origin_location: Optional[GeoLocation] = None
    destination_location: Optional[GeoLocation] = None
    destination_label: str = ""
    places: list[Place] = Field(default_factory=list)
    itinerary: list[DayItinerary] = Field(default_factory=list)
    travel_options: list[TravelOption] = Field(default_factory=list)
    recommended_travel: Optional[TravelOption] = None
    hotels: list[HotelOption] = Field(default_factory=list)
    recommended_hotel: Optional[HotelOption] = None
    agent_results: list[AgentResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    estimated_cost_usd: Optional[float] = None
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
    weather: list[DayWeather] = Field(default_factory=list)
    bookings: list[BookingItem] = Field(default_factory=list)
    supervisor: Optional[SupervisorDecision] = None
    supervisor_trace: list[ReasoningStep] = Field(default_factory=list)
    review_history: list[ReviewRecord] = Field(default_factory=list)
    status: str = Field(default="draft", description="draft, awaiting_approval, approved, or rejected")
