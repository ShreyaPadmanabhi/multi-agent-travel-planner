"""Optional LLM layer that adds practical guidance to an already-scheduled trip.

The schedule itself is computed deterministically. The model only writes advice
for slots that already exist, so a missing key or a failed call degrades the
copy rather than the plan.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import settings
from app.models.travel import DayItinerary, TravelPlanRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an experienced local travel guide.
For each itinerary item you are given, write one short practical tip (max 18 words):
booking advice, best entrance, what to see first, or timing warnings.
Also write a short evocative theme (max 6 words) for each day.
Return strict JSON of the form:
{"days": [{"day": 1, "theme": "...", "tips": [{"place_name": "...", "note": "..."}]}]}
Only reference the places you are given. Never invent attractions."""


class LLMService:
    def __init__(self) -> None:
        self._client = None
        if settings.llm_enabled:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            except Exception as exc:
                logger.warning("OpenAI client unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def narrate_itinerary(
        self, request: TravelPlanRequest, days: list[DayItinerary]
    ) -> list[DayItinerary] | None:
        if not self._client or not days:
            return None

        payload = {
            "destination": request.destination,
            "interests": request.interests,
            "days": [
                {
                    "day": day.day,
                    "places": [
                        item.place_name
                        for item in day.items
                        if item.activity.startswith("Visit ")
                    ],
                }
                for day in days
            ],
        }

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.4,
                ),
                timeout=45,
            )
            data = json.loads(response.choices[0].message.content or "{}")
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Itinerary narration skipped: %s", exc)
            return None

        tips: dict[tuple[int, str], str] = {}
        themes: dict[int, str] = {}
        for entry in data.get("days", []):
            day_number = entry.get("day")
            if not isinstance(day_number, int):
                continue
            if theme := entry.get("theme"):
                themes[day_number] = str(theme)[:60]
            for tip in entry.get("tips", []):
                place = tip.get("place_name")
                note = tip.get("note")
                if place and note:
                    tips[(day_number, place)] = str(note)[:200]

        enriched: list[DayItinerary] = []
        for day in days:
            items = [
                item.model_copy(update={"notes": tips.get((day.day, item.place_name), item.notes)})
                for item in day.items
            ]
            enriched.append(
                day.model_copy(update={"items": items, "theme": themes.get(day.day, day.theme)})
            )
        return enriched
