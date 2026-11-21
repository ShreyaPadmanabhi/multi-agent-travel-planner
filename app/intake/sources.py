"""Pull raw text and images out of each kind of trip material.

Each extractor returns an `Extraction`: the text it could recover, labelled by
method, plus a few images for the vision model to read when text alone is not
enough (scanned pages, pamphlets, on-screen captions in reels).

Downloaded media goes into a fresh temporary directory per request and is
only ever read as data, never executed.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from app.config import settings
from app.intake import models
from app.models.intake import SourceReport

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md"}

# A PDF page with less text than this is treated as scanned and OCR'd.
SCANNED_PAGE_CHARS = 40
MAX_VISION_IMAGES = 6


@dataclass
class Extraction:
    report: SourceReport
    texts: list[tuple[str, str]] = field(default_factory=list)  # (method, text)
    images: list[Image.Image] = field(default_factory=list)

    def add(self, method: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.texts.append((method, text))
            if method not in self.report.methods:
                self.report.methods.append(method)
            self.report.characters += len(text)

    def warn(self, message: str) -> None:
        logger.info("%s: %s", self.report.reference, message)
        self.report.warnings.append(message)


# --------------------------------------------------------------------- OCR

_ocr_engine = None


def ocr_image(image: Image.Image) -> str:
    """Read printed text from an image with RapidOCR (ONNX, no system install)."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()

    import numpy as np

    result, _ = _ocr_engine(np.array(image.convert("RGB")))
    if not result:
        return ""
    # Each entry is (box, text, confidence); keep reading order top to bottom.
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    return "\n".join(text for _, text, score in lines if float(score) >= 0.5)


# ------------------------------------------------------------------- media


def sample_frames(video_path: Path, count: int) -> list[Image.Image]:
    """Evenly spaced frames, skipping the first and last few percent (intros, end cards)."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        frames = []
        for i in range(count):
            position = int(total * (0.05 + 0.9 * (i + 0.5) / count))
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((1024, 1024))
                frames.append(image)
        return frames
    finally:
        capture.release()


async def extract_media_file(path: Path, report: SourceReport, extraction: Extraction | None = None) -> Extraction:
    """Video or audio: Whisper for speech, OCR and vision for what is on screen."""
    extraction = extraction or Extraction(report)
    size_mb = path.stat().st_size / 1_000_000

    if not models.available():
        extraction.warn("Speech transcription needs OPENAI_API_KEY (Whisper).")
    elif size_mb > settings.intake_max_media_mb:
        extraction.warn(
            f"Media is {size_mb:.0f} MB; Whisper accepts up to {settings.intake_max_media_mb} MB, so speech was skipped."
        )
    else:
        try:
            extraction.add("whisper", await models.transcribe(path))
        except Exception as exc:
            extraction.warn(f"Transcription failed: {type(exc).__name__}: {exc}")

    if path.suffix.lower() in VIDEO_EXTENSIONS:
        frames = sample_frames(path, settings.intake_video_frames)
        on_screen = [ocr_image(frame) for frame in frames]
        extraction.add("ocr", "\n".join(t for t in on_screen if t))
        extraction.images.extend(frames)
    return extraction


# ----------------------------------------------------------------- YouTube


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0] or None
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            return (parse_qs(parsed.query).get("v") or [None])[0]
        match = re.match(r"^/(?:shorts|embed|live)/([\w-]{6,})", parsed.path)
        if match:
            return match.group(1)
    return None


def _fetch_captions(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcripts = api.list(video_id)
    try:
        transcript = transcripts.find_transcript(["en", "en-GB", "en-US"])
    except Exception:
        # Any language will do: the extraction model reads it and answers in English.
        transcript = next(iter(transcripts))
    return " ".join(snippet.text for snippet in transcript.fetch())


def _download(url: str, workdir: Path, audio_only: bool) -> tuple[dict, Path | None]:
    import yt_dlp

    limit = settings.intake_max_media_mb * 1_000_000
    # Pre-muxed formats only: merging separate streams would need ffmpeg.
    fmt = (
        f"bestaudio[ext=m4a][filesize<{limit}]/bestaudio[filesize<{limit}]/worstaudio"
        if audio_only
        else f"best[ext=mp4][height<=720][filesize<{limit}]/best[ext=mp4][height<=720]/best[height<=720]"
    )
    options = {
        "format": fmt,
        "outtmpl": str(workdir / "media.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": limit,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads and downloads[0].get("filepath") else None
    return info, path if path and path.exists() else None


async def extract_video_url(url: str) -> Extraction:
    """YouTube, Shorts, Instagram reels, TikTok and anything else yt-dlp supports."""
    import asyncio

    video_id = youtube_video_id(url)
    report = SourceReport(kind="youtube" if video_id else "video", reference=url)
    extraction = Extraction(report)

    if video_id:
        try:
            extraction.add("captions", await asyncio.to_thread(_fetch_captions, video_id))
        except Exception as exc:
            extraction.warn(f"No captions available ({type(exc).__name__}); falling back to audio.")

    # Captions cover speech; reels and caption-less videos still need the media.
    need_speech = "captions" not in report.methods
    want_frames = not video_id
    if not (need_speech or want_frames):
        try:
            info = await asyncio.to_thread(_metadata, url)
            _add_metadata(extraction, info)
        except Exception as exc:
            extraction.warn(f"Could not read video metadata: {type(exc).__name__}")
        return extraction

    with tempfile.TemporaryDirectory(prefix="intake-") as tmp:
        try:
            info, path = await asyncio.to_thread(_download, url, Path(tmp), not want_frames)
        except Exception as exc:
            extraction.warn(f"Could not download the video: {type(exc).__name__}: {str(exc)[:200]}")
            return extraction
        _add_metadata(extraction, info)
        if path is None:
            extraction.warn("The video downloaded without a usable media file.")
            return extraction
        await extract_media_file(path, report, extraction)
    return extraction


def _metadata(url: str) -> dict:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False)


def _add_metadata(extraction: Extraction, info: dict) -> None:
    # Creators often list every stop, hotel and price in the description.
    parts = [info.get("title") or "", info.get("description") or ""]
    extraction.add("description", "\n".join(p for p in parts if p))


# --------------------------------------------------------- documents/images


def extract_pdf(path: Path, reference: str) -> Extraction:
    import pypdfium2 as pdfium
    from pypdf import PdfReader

    extraction = Extraction(SourceReport(kind="pdf", reference=reference))
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    limit = min(page_count, settings.intake_max_pdf_pages)
    if page_count > limit:
        extraction.warn(f"Only the first {limit} of {page_count} pages were read.")

    scanned: list[int] = []
    text_pages = []
    for index in range(limit):
        text = reader.pages[index].extract_text() or ""
        if len(text.strip()) < SCANNED_PAGE_CHARS:
            scanned.append(index)
        else:
            text_pages.append(text)
    extraction.add("pdf-text", "\n\n".join(text_pages))

    if scanned:
        document = pdfium.PdfDocument(str(path))
        try:
            ocr_pages = []
            for index in scanned:
                image = document[index].render(scale=2).to_pil()
                ocr_pages.append(ocr_image(image))
                if len(extraction.images) < MAX_VISION_IMAGES:
                    image.thumbnail((1400, 1400))
                    extraction.images.append(image)
            extraction.add("ocr", "\n\n".join(ocr_pages))
        finally:
            document.close()
    return extraction


def extract_image(path: Path, reference: str) -> Extraction:
    extraction = Extraction(SourceReport(kind="image", reference=reference))
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    extraction.add("ocr", ocr_image(image))
    image.thumbnail((1400, 1400))
    extraction.images.append(image)
    return extraction


def extract_text(content: str, reference: str) -> Extraction:
    extraction = Extraction(SourceReport(kind="text", reference=reference))
    extraction.add("text", content)
    return extraction


async def extract_file(path: Path, reference: str | None = None) -> Extraction:
    reference = reference or path.name
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, reference)
    if suffix in IMAGE_EXTENSIONS:
        return extract_image(path, reference)
    if suffix in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        kind = "video" if suffix in VIDEO_EXTENSIONS else "audio"
        return await extract_media_file(path, SourceReport(kind=kind, reference=reference))
    if suffix in TEXT_EXTENSIONS:
        return extract_text(path.read_text(encoding="utf-8", errors="replace"), reference)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
