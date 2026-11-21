"""Real road distances and durations from OSRM (keyless) plus haversine fallback."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from app.config import settings
from app.providers.http import ProviderError, request_json

# Typical walking speed in a city, including crossings and detours.
WALKING_KMH = 4.5


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    earth_radius_km = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(a))


class RouteResult:
    def __init__(self, distance_km: float, duration_minutes: int, estimated: bool) -> None:
        self.distance_km = distance_km
        self.duration_minutes = duration_minutes
        self.estimated = estimated


class RoutingProvider:
    source = "OSRM (OpenStreetMap routing)"

    async def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        profile: str = "driving",
    ) -> RouteResult:
        lat1, lon1 = origin
        lat2, lon2 = destination

        try:
            payload = await request_json(
                "GET",
                f"{settings.osrm_url}/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}",
                params={"overview": "false"},
                limiter="osrm",
                retries=1,
            )
            routes = payload.get("routes") or []
            if payload.get("code") == "Ok" and routes:
                best = routes[0]
                return RouteResult(
                    distance_km=round(best["distance"] / 1000, 1),
                    duration_minutes=int(round(best["duration"] / 60)),
                    estimated=False,
                )
        except ProviderError:
            pass

        # Straight-line fallback keeps the itinerary usable if OSRM is unreachable.
        straight = haversine_km(lat1, lon1, lat2, lon2)
        return RouteResult(
            distance_km=round(straight, 1),
            duration_minutes=int(round(straight / WALKING_KMH * 60)),
            estimated=True,
        )

    async def walking_minutes(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> tuple[int, float]:
        """Walking time and distance between two city points."""
        result = await self.route(origin, destination, profile="foot")
        if result.estimated:
            # OSRM's public demo server only hosts the driving profile, so a
            # constant-speed walking estimate is the honest fallback here.
            distance = haversine_km(*origin, *destination) * 1.25
            return int(round(distance / WALKING_KMH * 60)), round(distance, 2)
        return result.duration_minutes, result.distance_km
