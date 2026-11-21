"""Builds a day-by-day schedule from real coordinates.

The scheduling problem has two parts. First, decide which places belong on the
same day — a traveller should not criss-cross the city, so places are grouped
geographically with k-means over their coordinates. Second, decide the order
within a day, which is a small travelling-salesman problem solved with a
nearest-neighbour heuristic plus 2-opt improvement.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from app.agents.base import AgentSkipped, BaseAgent
from app.models.travel import DayItinerary, ItineraryItem, Place, TravelPlan
from app.providers.routing import haversine_km
from app.services.llm_service import LLMService

# Average city walking speed including crossings, plus a detour factor to
# approximate street distance from straight-line distance.
WALKING_KMH = 4.5
STREET_DETOUR_FACTOR = 1.3
# Beyond this, walking stops being reasonable and we assume transit or a taxi.
WALK_LIMIT_KM = 2.0
TRANSIT_KMH = 18.0

# `max_places` only caps how many candidates a day may hold; the real limit is
# `max_hours`, so short visits naturally allow more stops than long ones.
PACE_SETTINGS = {
    "relaxed": {"max_hours": 5.0, "max_places": 3, "start": 10.0},
    "balanced": {"max_hours": 7.5, "max_places": 4, "start": 9.5},
    "packed": {"max_hours": 10.0, "max_places": 6, "start": 9.0},
}

LUNCH_HOUR = 13.0
LUNCH_DURATION = 1.0


def _format_clock(hours: float) -> str:
    hour = int(hours) % 24
    minute = int(round((hours - int(hours)) * 60))
    if minute == 60:
        hour, minute = (hour + 1) % 24, 0
    return f"{hour:02d}:{minute:02d}"


def _travel_estimate(a: Place, b: Place) -> tuple[int, str, float]:
    """Return (minutes, mode, distance_km) between two places."""
    straight = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
    distance = straight * STREET_DETOUR_FACTOR
    if distance <= WALK_LIMIT_KM:
        return max(int(round(distance / WALKING_KMH * 60)), 5), "walk", distance
    return max(int(round(distance / TRANSIT_KMH * 60)), 8), "transit", distance


class ItineraryAgent(BaseAgent):
    name = "ItineraryAgent"
    description = "Groups places by area and schedules each day with real travel times."
    source = "Geographic clustering + nearest-neighbour routing"

    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or LLMService()

    async def run(self, plan: TravelPlan) -> TravelPlan:
        if not plan.places:
            raise AgentSkipped("No places available to schedule.")

        request = plan.request
        pace = PACE_SETTINGS.get(request.pace, PACE_SETTINGS["balanced"])
        num_days = request.days

        clusters = self._cluster_by_day(plan.places, num_days, int(pace["max_places"]))

        itinerary: list[DayItinerary] = []
        for index, cluster in enumerate(clusters):
            day_date = request.start_date + timedelta(days=index)
            ordered = self._order_route(cluster)
            day = self._schedule_day(
                day_number=index + 1,
                day_date=day_date,
                places=ordered,
                pace=pace,
                is_first_day=index == 0,
                is_last_day=index == num_days - 1 and num_days > 1,
            )
            itinerary.append(day)

        plan.itinerary = await self.llm.narrate_itinerary(request, itinerary) or itinerary
        return plan

    def _cluster_by_day(
        self, places: list[Place], num_days: int, max_per_day: int
    ) -> list[list[Place]]:
        """Partition places into one geographic cluster per day.

        Capacity is applied after clustering: k-means optimises for compactness
        and has no notion of how many stops fit in a day, so overflowing days
        hand their least notable places to the emptiest day.
        """
        capacity = max_per_day * num_days
        # Requested places always make the cut, ahead of discovered ones.
        ordered = [p for p in places if p.pinned] + [p for p in places if not p.pinned]
        selected = ordered[: max(capacity, sum(p.pinned for p in places))]

        if num_days == 1 or len(selected) <= 1:
            return [selected]

        clusters = self._kmeans(selected, num_days)
        return self._balance(clusters, max_per_day)

    @staticmethod
    def _kmeans(places: list[Place], k: int, iterations: int = 25) -> list[list[Place]]:
        k = min(k, len(places))
        # Seed deterministically so the same request yields the same plan.
        rng = random.Random(len(places) * 31 + k)

        # k-means++ style seeding: spread initial centroids apart so clusters
        # do not collapse onto the same neighbourhood.
        centroids = [(places[0].latitude, places[0].longitude)]
        while len(centroids) < k:
            distances = [
                min(haversine_km(p.latitude, p.longitude, cy, cx) for cy, cx in centroids)
                for p in places
            ]
            total = sum(distances)
            if total == 0:
                centroids.append((places[rng.randrange(len(places))].latitude,
                                  places[rng.randrange(len(places))].longitude))
                continue
            threshold = rng.random() * total
            running = 0.0
            for place, distance in zip(places, distances):
                running += distance
                if running >= threshold:
                    centroids.append((place.latitude, place.longitude))
                    break

        assignments = [0] * len(places)
        for _ in range(iterations):
            changed = False
            for i, place in enumerate(places):
                best = min(
                    range(len(centroids)),
                    key=lambda c: haversine_km(
                        place.latitude, place.longitude, centroids[c][0], centroids[c][1]
                    ),
                )
                if best != assignments[i]:
                    assignments[i] = best
                    changed = True

            for c in range(len(centroids)):
                members = [p for p, a in zip(places, assignments) if a == c]
                if members:
                    centroids[c] = (
                        sum(p.latitude for p in members) / len(members),
                        sum(p.longitude for p in members) / len(members),
                    )

            if not changed:
                break

        clusters = [[p for p, a in zip(places, assignments) if a == c] for c in range(len(centroids))]
        clusters = [c for c in clusters if c]

        # Visit the busiest area first; travellers front-load the big sights.
        clusters.sort(key=lambda c: -sum(p.popularity for p in c))
        return clusters

    @staticmethod
    def _balance(clusters: list[list[Place]], max_per_day: int) -> list[list[Place]]:
        clusters = [list(c) for c in clusters]
        for _ in range(len(clusters) * 3):
            overfull = [c for c in clusters if len(c) > max_per_day]
            if not overfull:
                break
            donor = max(clusters, key=len)
            receiver = min(clusters, key=len)
            if len(donor) - len(receiver) <= 1:
                break
            donor.sort(key=lambda p: (-p.pinned, -p.popularity))
            receiver.append(donor.pop())
        return clusters

    @staticmethod
    def _order_route(places: list[Place]) -> list[Place]:
        """Nearest-neighbour ordering refined with 2-opt segment reversal."""
        if len(places) <= 2:
            return places

        remaining = places[:]
        # Start from the most notable place: it is the anchor of the day and is
        # best visited before crowds and fatigue build up.
        route = [max(remaining, key=lambda p: p.popularity)]
        remaining.remove(route[0])

        while remaining:
            last = route[-1]
            nearest = min(
                remaining,
                key=lambda p: haversine_km(last.latitude, last.longitude, p.latitude, p.longitude),
            )
            route.append(nearest)
            remaining.remove(nearest)

        def route_length(order: list[Place]) -> float:
            return sum(
                haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
                for a, b in zip(order, order[1:])
            )

        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                    if route_length(candidate) < route_length(route) - 1e-9:
                        route = candidate
                        improved = True
        return route

    def _schedule_day(
        self,
        day_number: int,
        day_date: date,
        places: list[Place],
        pace: dict,
        is_first_day: bool,
        is_last_day: bool,
    ) -> DayItinerary:
        clock = float(pace["start"])
        budget_hours = float(pace["max_hours"])
        spent = 0.0
        items: list[ItineraryItem] = []
        walking_km = 0.0
        lunch_taken = False

        if is_first_day:
            items.append(
                ItineraryItem(
                    start_time=_format_clock(clock),
                    end_time=_format_clock(clock + 1.0),
                    activity="Arrive and check in",
                    place_name="Accommodation",
                    notes="Drop bags before starting the day.",
                )
            )
            clock += 1.0

        previous: Place | None = None
        for place in places:
            # Checked before travel is added, so a skipped stop costs no time.
            # Requested places are kept even when they overrun the day.
            duration = place.estimated_hours
            if spent + duration > budget_hours and not place.pinned:
                continue

            if previous is not None:
                minutes, mode, distance = _travel_estimate(previous, place)
                clock += minutes / 60
                if mode == "walk":
                    walking_km += distance
            else:
                minutes, mode = 0, ""

            if not lunch_taken and clock >= LUNCH_HOUR:
                items.append(
                    ItineraryItem(
                        start_time=_format_clock(clock),
                        end_time=_format_clock(clock + LUNCH_DURATION),
                        activity="Lunch break",
                        place_name="Nearby restaurant",
                        notes="Pick somewhere close to the next stop.",
                    )
                )
                clock += LUNCH_DURATION
                lunch_taken = True

            items.append(
                ItineraryItem(
                    start_time=_format_clock(clock),
                    end_time=_format_clock(clock + duration),
                    activity=f"Visit {place.name}",
                    place_name=place.name,
                    notes=place.description[:160],
                    travel_minutes_from_previous=minutes,
                    travel_mode=mode,
                )
            )
            clock += duration
            spent += duration
            previous = place

        if is_last_day:
            items.append(
                ItineraryItem(
                    start_time=_format_clock(max(clock, 17.0)),
                    end_time=_format_clock(max(clock, 17.0) + 1.0),
                    activity="Depart",
                    place_name="Head to station or airport",
                    notes="Allow buffer time for check-in.",
                )
            )

        return DayItinerary(
            day=day_number,
            date=day_date,
            theme=self._theme(places),
            items=items,
            walking_distance_km=round(walking_km, 1),
        )

    @staticmethod
    def _theme(places: list[Place]) -> str:
        if not places:
            return "Free day"
        counts: dict[str, int] = {}
        for place in places:
            counts[place.category] = counts.get(place.category, 0) + 1
        dominant = max(counts, key=counts.get)
        labels = {
            "museum": "Museums & galleries",
            "historic": "Historic sites",
            "monument": "Monuments & landmarks",
            "religious": "Sacred architecture",
            "landmark": "City icons",
            "viewpoint": "Views over the city",
            "park": "Parks & green space",
            "nature": "Nature & outdoors",
            "food": "Food & markets",
            "entertainment": "Arts & entertainment",
            "sports": "Sports & venues",
            "attraction": "Highlights",
        }
        return labels.get(dominant, "Highlights")

    def success_message(self, plan: TravelPlan) -> str:
        stops = sum(1 for day in plan.itinerary for item in day.items if item.travel_mode or "Visit" in item.activity)
        return f"Scheduled {len(plan.itinerary)} days covering {stops} stops."
