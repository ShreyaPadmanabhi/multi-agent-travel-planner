"""OpenAI calls used by intake: Whisper, GPT-4o vision, and schema extraction."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from app.config import settings
from app.models.intake import TripIntake
from app.providers.places import INTEREST_CATEGORIES

logger = logging.getLogger(__name__)

_client = None


def _openai():
    global _client
    if _client is None and settings.llm_enabled:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def available() -> bool:
    return _openai() is not None


async def transcribe(path: Path) -> str:
    with path.open("rb") as handle:
        response = await _openai().audio.transcriptions.create(
            model=settings.openai_transcription_model, file=handle
        )
    return response.text


VISION_PROMPT = """These images come from travel material: video frames, a scanned itinerary, or a pamphlet.
Transcribe every travel-relevant detail you can read or recognise: place and attraction names,
hotels, restaurants, day-by-day structure, dates, prices, transport, opening hours.
If you recognise a landmark without a caption, name it and say it is recognised, not captioned.
Plain text only, no commentary."""


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


async def read_images(images: list[Image.Image]) -> str:
    content: list[dict] = [{"type": "text", "text": VISION_PROMPT}]
    content += [{"type": "image_url", "image_url": {"url": _data_url(i), "detail": "low"}} for i in images]
    response = await _openai().chat.completions.create(
        model=settings.openai_vision_model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _extraction_prompt() -> str:
    schema = json.dumps(TripIntake.model_json_schema())
    interests = ", ".join(sorted(INTEREST_CATEGORIES))
    return f"""You extract a structured trip plan from travel material (vlog transcripts, video
descriptions, OCR of pamphlets and itineraries, image descriptions).
Return JSON matching this JSON Schema exactly:
{schema}

Rules:
- destination is the main city or region the trip is based in.
- places: every specific, named, visitable place. Use kind=hotel for accommodation and
  kind=food for restaurants, cafes and markets. Use the most common English name.
  Set day when the material assigns the place to a day.
- interests: choose only from [{interests}].
- Only include facts stated in the material. Use null for anything not stated; never guess dates.
- budget_usd: convert only if a total budget is stated in another currency and the rate is obvious; else null."""


async def structure(text: str, hint: str = "") -> TripIntake:
    user = text if not hint else f"Traveller note: {hint}\n\n{text}"
    response = await _openai().chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _extraction_prompt()},
            {"role": "user", "content": user[:60_000]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content or "{}")
    try:
        return TripIntake.model_validate(raw)
    except ValidationError as exc:
        # Keep whatever validates rather than discarding the whole extraction.
        logger.warning("Intake schema validation failed, dropping bad fields: %s", exc)
        bad = {str(error["loc"][0]) for error in exc.errors() if error["loc"]}
        return TripIntake.model_validate({k: v for k, v in raw.items() if k not in bad})
