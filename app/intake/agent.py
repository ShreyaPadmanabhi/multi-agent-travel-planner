"""Multimodal intake agent: trip material in, `IntakeResult` out.

Inputs are URLs (YouTube, Shorts, reels, TikTok), uploaded files (PDF, images,
video, audio, text) or pasted text. Every source is reduced to text by the
cheapest method that works — captions before Whisper, embedded PDF text before
OCR — and images go to GPT-4o vision so landmarks and pamphlet layouts are read
too. One extraction call then maps the combined evidence onto `TripIntake`.

Without an OpenAI key, captions, PDF text and OCR still work, and a heuristic
pulls candidate place names; it needs the traveller to state the destination.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from app.intake import models
from app.intake.sources import Extraction, extract_file, extract_text, extract_video_url
from app.models.intake import IntakePlace, IntakeResult, TripIntake

MAX_EVIDENCE_CHARS = 60_000
MAX_HEURISTIC_PLACES = 12

# Capitalised words that start sentences or name things that are not places.
STOPWORDS = {
    "The", "This", "That", "These", "Then", "There", "And", "But", "So", "We", "I", "You", "It",
    "Our", "My", "Day", "Today", "Tomorrow", "Next", "First", "Finally", "After", "Before", "If",
    "Here", "What", "When", "Where", "Also", "Just", "Now", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December", "Subscribe",
    "Instagram", "YouTube", "TikTok", "Link", "Hi", "Hello", "Welcome", "Okay", "Yes", "No",
    "Breakfast", "Lunch", "Dinner", "Stay", "Discover", "Visit", "Explore", "Book", "Tickets",
}
SENTENCE_BREAK = re.compile(r"(?<!\bSt)(?<!\bSte)(?<!\bMt)(?<!\bDr)(?<!\bFt)[.!?;:]\s+")
HOTEL_WORDS = re.compile(r"\b(hotel|hostel|inn|resort|suites|lodge|b&b|guesthouse|ryokan)\b", re.I)
FOOD_WORDS = re.compile(r"\b(cafe|café|restaurant|trattoria|bistro|bar|market|bakery|osteria)\b", re.I)


class IntakeAgent:
    name = "IntakeAgent"

    async def process(
        self,
        urls: list[str] | None = None,
        files: list[tuple[Path, str]] | None = None,
        text: str = "",
        destination_hint: str = "",
    ) -> IntakeResult:
        extractions: list[Extraction] = []
        warnings: list[str] = []

        for url in urls or []:
            extractions.append(await extract_video_url(url.strip()))
        for path, reference in files or []:
            try:
                extractions.append(await extract_file(path, reference))
            except Exception as exc:
                warnings.append(f"{reference}: {type(exc).__name__}: {exc}")
        if text.strip():
            extractions.append(extract_text(text, "pasted text"))

        if not extractions:
            raise ValueError("Provide at least one URL, file or block of text.")

        if models.available():
            await self._read_images(extractions, warnings)

        evidence = self._combine(extractions)
        for extraction in extractions:
            warnings.extend(f"{extraction.report.reference}: {w}" for w in extraction.report.warnings)

        if not evidence.strip():
            warnings.append("No text could be recovered from the material.")
            trip, method = TripIntake(destination=destination_hint or None), "heuristic"
        elif models.available():
            trip, method = await models.structure(evidence, destination_hint), "llm"
        else:
            trip, method = self._heuristic(evidence, destination_hint), "heuristic"
            warnings.append(
                "No OPENAI_API_KEY: places were picked out heuristically. Review them before planning."
            )

        if destination_hint and not trip.destination:
            trip.destination = destination_hint

        return IntakeResult(
            trip=trip,
            sources=[e.report for e in extractions],
            extraction=method,
            warnings=warnings,
            evidence_excerpt=evidence[:1500],
        )

    @staticmethod
    async def _read_images(extractions: list[Extraction], warnings: list[str]) -> None:
        for extraction in extractions:
            if not extraction.images:
                continue
            try:
                extraction.add("vision", await models.read_images(extraction.images[:6]))
            except Exception as exc:
                warnings.append(f"{extraction.report.reference}: vision failed ({type(exc).__name__}).")

    @staticmethod
    def _combine(extractions: list[Extraction]) -> str:
        blocks = []
        for extraction in extractions:
            for method, text in extraction.texts:
                blocks.append(f"### {extraction.report.kind} {extraction.report.reference} [{method}]\n{text}")
        return "\n\n".join(blocks)[:MAX_EVIDENCE_CHARS]

    @staticmethod
    def _heuristic(evidence: str, destination_hint: str) -> TripIntake:
        """Frequency-ranked capitalised phrases as candidate places.

        Crude by design: the destination agent geocodes each candidate near the
        destination and drops the ones that do not resolve, so false positives
        cost a warning rather than a wrong stop.
        """
        body = "\n".join(line for line in evidence.splitlines() if not line.startswith("### "))
        body = re.sub(r"\S+\.(?:com|to|ly|net|org|io|me)\S*|[@#]\w+", " ", body)  # links, handles, tags
        destination = destination_hint.lower().split(",")[0].strip()
        # Phrases never span a line break: OCR puts separate items on separate lines.
        phrase = r"[A-Z][\w'’.-]+(?:[ \t]+(?:of|de|di|del|da|la|le|the)?[ \t]*[A-Z][\w'’.-]+){0,4}"
        counts: Counter[str] = Counter()
        # A full stop ends a name ("Market. We"), except in abbreviations ("St. Peter").
        sentences = [
            sentence
            for line in body.splitlines()
            for sentence in SENTENCE_BREAK.split(line.title() if line.isupper() else line)
        ]
        for sentence in sentences:
            for match in re.finditer(phrase, sentence):
                words = match.group(0).strip(" .-").split()
                while words and words[0] in STOPWORDS:
                    words = words[1:]
                while words and words[-1] in STOPWORDS:
                    words = words[:-1]
                candidate = re.sub(r"['’]s$", "", " ".join(words))
                if (
                    len(candidate) < 4
                    or candidate in STOPWORDS
                    or candidate.lower() == destination
                    or any(ch.isdigit() for ch in candidate)
                ):
                    continue
                counts[candidate] += 1

        def is_venue(name: str) -> bool:
            return bool(HOTEL_WORDS.search(name) or FOOD_WORDS.search(name))

        # A single capitalised word is usually just a sentence start, so it needs
        # repeat mentions; multi-word names and named venues count from one.
        candidates = [c for c in counts if " " in c or counts[c] >= 2 or is_venue(c)]
        # OCR sometimes drops spaces ("StayatHotel Artemide"); keep the clean variant.
        candidates = [c for c in candidates if not any(o != c and c.endswith(o) for o in candidates)]
        ranked = sorted(candidates, key=lambda c: (-is_venue(c), -counts[c] * (2 if " " in c else 1), c))
        places = []
        for name in ranked[:MAX_HEURISTIC_PLACES]:
            kind = "hotel" if HOTEL_WORDS.search(name) else "food" if FOOD_WORDS.search(name) else "attraction"
            places.append(IntakePlace(name=name, kind=kind))

        duration = None
        # No leading \b: OCR often glues the number to the previous word ("in3 Days").
        if match := re.search(r"(?<!\d)(\d{1,2})[\s-]*(?:days?|day trip)\b", body, re.I):
            duration = min(max(int(match.group(1)), 1), 30)

        return TripIntake(destination=destination_hint or None, duration_days=duration, places=places)
