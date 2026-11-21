"""MCP server exposing the travel data providers as tools.

The supervisor's ReAct loop reaches live data through this server rather than
importing providers directly, so any MCP client (Claude Desktop, an IDE,
another agent) can use the same tools:

    python -m app.mcp.server            # stdio transport
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from mcp.server.mcpserver import MCPServer

from app.providers.flights import FlightProvider
from app.providers.geocoding import GeocodingProvider
from app.providers.hotels import HotelProvider
from app.providers.places import PlacesProvider
from app.providers.routing import RoutingProvider, haversine_km
from app.providers.weather import WeatherProvider

# stdout carries the JSON-RPC stream, so all logging must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

server = MCPServer("travelplanner")

geocoder = GeocodingProvider()
places = PlacesProvider()
routing = RoutingProvider()
flights = FlightProvider()
hotels = HotelProvider()
weather = WeatherProvider()


@server.tool()
async def geocode(query: str) -> dict:
    """Resolve a city, landmark or address to coordinates and a country code."""
    location = await geocoder.geocode(query)
    return location.model_dump()


@server.tool()
async def search_places(destination: str, interests: list[str] | None = None, limit: int = 12) -> dict:
    """Find notable attractions in a destination, ranked by Wikipedia coverage and interests."""
    location = await geocoder.geocode(destination)
    found = await places.find_places(location, interests or [], limit=min(max(limit, 1), 30))
    return {
        "destination": location.display_name,
        "places": [
            {
                "name": p.name,
                "category": p.category,
                "popularity": p.popularity,
                "estimated_hours": p.estimated_hours,
                "fee_required": p.fee_required,
            }
            for p in found
        ],
    }


@server.tool()
async def find_route(origin: str, destination: str) -> dict:
    """Road distance and driving time between two places, or a flag that no road route exists."""
    a = await geocoder.geocode(origin)
    b = await geocoder.geocode(destination)
    route = await routing.route(a.coordinates, b.coordinates)
    return {
        "origin": a.short_name,
        "destination": b.short_name,
        "straight_line_km": round(haversine_km(*a.coordinates, *b.coordinates), 1),
        "road_route_found": not route.estimated,
        "road_distance_km": None if route.estimated else route.distance_km,
        "driving_minutes": None if route.estimated else route.duration_minutes,
    }


@server.tool()
async def search_flights(origin: str, destination: str, departure_date: str, travelers: int = 1) -> dict:
    """Flight offers from Duffel (needs DUFFEL_ACCESS_TOKEN). departure_date is YYYY-MM-DD."""
    if not flights.enabled:
        return {"available": False, "reason": "Duffel access token is not configured."}
    a = await geocoder.geocode(origin)
    b = await geocoder.geocode(destination)
    offers = await flights.search(a, b, date.fromisoformat(departure_date), travelers, max_results=5)
    return {
        "available": True,
        "offers": [
            {
                "provider": o.provider,
                "price_usd": o.price_usd,
                "duration_minutes": o.duration_minutes,
                "stops": o.stops,
            }
            for o in offers
        ],
    }


@server.tool()
async def search_hotels(destination: str, check_in: str, check_out: str, travelers: int = 1) -> dict:
    """Real, mapped accommodation near the destination centre (no nightly rates)."""
    location = await geocoder.geocode(destination)
    found = await hotels.search(
        location, date.fromisoformat(check_in), date.fromisoformat(check_out), travelers, limit=8
    )
    return {
        "hotels": [
            {
                "name": h.name,
                "stars": h.stars,
                "total_price_usd": h.total_price_usd,
                "distance_to_center_km": h.distance_to_center_km,
            }
            for h in found
        ],
    }


@server.tool()
async def get_weather(destination: str, start_date: str, end_date: str) -> dict:
    """Daily weather for a date range: a forecast within ~15 days, otherwise last year's same dates."""
    location = await geocoder.geocode(destination)
    days = await weather.daily(location, date.fromisoformat(start_date), date.fromisoformat(end_date))
    return {"destination": location.short_name, "days": [d.model_dump(mode="json") for d in days]}


if __name__ == "__main__":
    server.run("stdio")
