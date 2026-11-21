"""Run the intake agent on generated material, and optionally on real URLs or files.

    python scripts/test_intake.py                      # synthetic pamphlet image + scanned PDF
    python scripts/test_intake.py <url-or-file> ...    # real inputs as well
"""

import asyncio
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from app.intake import IntakeAgent

PAMPHLET = [
    "DISCOVER ROME IN 3 DAYS",
    "Day 1: Colosseum and Roman Forum",
    "Day 2: Vatican Museums, St. Peter's Basilica",
    "Day 3: Trevi Fountain, Pantheon",
    "Stay at Hotel Artemide",
    "Dinner at Trattoria Da Enzo",
]


def make_pamphlet(path: pathlib.Path) -> Image.Image:
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default(size=44)
    for i, line in enumerate(PAMPHLET):
        draw.text((60, 50 + i * 100), line, fill="black", font=font)
    image.save(path)
    return image


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        image = make_pamphlet(tmp_path / "pamphlet.png")
        # Saving an image as PDF yields a page with no text layer: the scanned path.
        image.save(tmp_path / "scanned.pdf")

        files = [(tmp_path / "pamphlet.png", "pamphlet.png"), (tmp_path / "scanned.pdf", "scanned.pdf")]
        urls = []
        for arg in sys.argv[1:]:
            if arg.startswith("http"):
                urls.append(arg)
            else:
                files.append((pathlib.Path(arg), pathlib.Path(arg).name))

        result = await IntakeAgent().process(urls=urls, files=files, destination_hint="Rome")

    print("EXTRACTION:", result.extraction)
    for source in result.sources:
        print(f"  {source.kind:8} {source.reference[:50]:52} methods={source.methods} chars={source.characters}")
        for warning in source.warnings:
            print(f"           ! {warning}")
    trip = result.trip
    print("DESTINATION:", trip.destination, "| DAYS:", trip.duration_days)
    for place in trip.places:
        print(f"  {place.kind:10} {place.name}")
    request = result.to_request(origin="London")
    print("REQUEST must_visit:", request.must_visit)
    print("REQUEST preferred_hotels:", request.preferred_hotels)
    print("REQUEST dates:", request.start_date, "->", request.end_date)
    print("WARNINGS:", result.warnings)


asyncio.run(main())
