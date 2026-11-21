# TravelPlanner — Multi-Agent Travel Planning System

A LangGraph-based multi-agent system that plans real trips from live data: it
discovers actual attractions, schedules them geographically, finds real routes,
and searches real accommodation. No mock data and no invented prices — when a
figure isn't available from a provider, the system says so instead of making
one up.

## Agents

| Stage | Agent | What it does | Data source |
|------|-------|--------------|-------------|
| Intake | **IntakeAgent** | Turns YouTube vlogs, reels, PDFs, pamphlet photos and pasted text into one `TripIntake` schema | YouTube captions, Whisper, GPT-4o vision, RapidOCR, pypdf |
| 1 | **DestinationAgent** | Geocodes origin and destination, finds notable attractions, pins places named in your material | Nominatim, Wikidata, Wikipedia |
| 2 | **Supervisor** | ReAct loop over MCP tools; decides which specialists run and, on revisions, what changed | OpenAI + the project MCP server |
| 3 | **ItineraryAgent** | Clusters attractions by area per day and orders each day into an efficient route | Computed locally |
| 3 | **TransportAgent** | Searches flight offers and computes real road routes | Duffel, OSRM |
| 3 | **HotelAgent** | Finds real accommodation near the centre | OpenStreetMap |
| 3 | **WeatherAgent** | Daily forecast, or last year's weather beyond the forecast window | Open-Meteo |
| 4 | **You** | Approve, request changes, or reject; the graph pauses until you answer | — |
| 5 | **BookingAgent** | Prefilled booking links for the approved plan (no payment taken) | Provider deep links |

The four stage-3 specialists depend only on discovery, never on each other, so
the supervisor dispatches them together and LangGraph runs them in one
superstep.

```
 YouTube / reel / PDF / pamphlet ──► IntakeAgent ──► TripIntake ──► TravelPlanRequest
                                                                        │
                                       ┌──────────── supervisor ◄───────┘
                                       │  (ReAct: Thought → MCP tool → Observation …)
                                       ▼
                                   discover ──► supervisor
                                                   │ dispatch (parallel)
                        ┌───────────────┬──────────┴────┬───────────────┐
                        ▼               ▼               ▼               ▼
                    itinerary       transport         hotel          weather
                        └───────────────┴───────┬───────┴───────────────┘
                                                ▼
                                            finalize
                                                ▼
                                  human_review  ── interrupt() ── waits for you
                                  │ approve          │ revise            │ reject
                                  ▼                  ▼                   ▼
                               booking ─► END    supervisor             END
                                                 (re-runs only the affected specialists)
```

## Supervisor, ReAct and MCP

`app/mcp/server.py` is a standalone MCP server exposing the providers as six
tools: `geocode`, `search_places`, `find_route`, `search_flights`,
`search_hotels`, `get_weather`. Any MCP client can use it:

```powershell
.venv\Scripts\python -m app.mcp.server
```

With `OPENAI_API_KEY` set, the supervisor launches that server over stdio,
offers its tools to the model, and runs a ReAct loop: the model writes a
thought, calls a tool, reads the observation, and repeats until it calls
`dispatch_specialists`. Every step is recorded in `plan.supervisor_trace` and
shown in the UI.

The model proposes and the rules decide. A field-impact table
(`FIELD_IMPACT` in `app/agents/supervisor.py`) maps every request field to
the specialists it invalidates. Whatever the model returns, those specialists
re-run, invalid changes are dropped by validation, and the first pass never
changes your request. Without a key, or if the loop fails, the same rules
decide alone and keyword rules interpret your feedback.

## Human-in-the-loop review

The graph is compiled with a checkpointer. `human_review` calls `interrupt()`,
so a draft waits in the checkpoint until you answer:

- **approve**: the booking agent runs and the plan becomes `approved`.
- **revise**: free-text feedback and/or explicit field edits go back to the
  supervisor. Changing the pace re-runs only the itinerary. The other three
  branches are reused from the checkpoint.
- **reject**: the graph ends.

```
POST /api/plan                      -> {thread_id, status: "awaiting_approval", plan}
POST /api/plan/{thread_id}/review   {"action": "revise", "feedback": "too packed, add Villa Borghese"}
POST /api/plan/{thread_id}/review   {"action": "approve"}
GET  /api/plan/{thread_id}
```

`POST /api/plan?auto_approve=true` skips the pause for scripted use. The
checkpointer is in-memory; swap in a SQLite or Postgres saver to keep paused
plans across restarts.

## Multimodal intake

`POST /api/intake` (multipart: `urls`, `files`, `text`, `destination_hint`,
`origin`), or the **Import trip material** panel in Streamlit.

| Input | How text is recovered |
|-------|-----------------------|
| YouTube / Shorts | Captions first (any language); title and description; no captions → audio download → Whisper |
| Reels, TikTok, other video links | yt-dlp download → Whisper for speech, sampled frames → OCR + GPT-4o vision |
| Video / audio files | Whisper; frames → OCR + vision |
| PDF | Embedded text; pages without a text layer are rendered and OCR'd, and sent to vision |
| Images / pamphlets | RapidOCR (ONNX, no system install) + GPT-4o vision |
| Pasted text | As is |

All evidence is combined and mapped onto `TripIntake` (`app/models/intake.py`)
by one extraction call, then validated. Fields that fail validation are dropped
individually instead of discarding the whole result. `IntakeResult.to_request`
feeds the planner: named attractions become pinned **must-visit** stops, which
the itinerary agent never drops, and named hotels become **preferred stays**,
which the booking agent links first.

Without a key, captions, PDF text and OCR still work. A heuristic picks
candidate place names, and the destination agent geocodes each one near the
destination and drops any that don't resolve. Whisper is limited to 25 MB per
file; downloads are pre-muxed formats, so no ffmpeg is needed.

## Measuring the speed-up

```powershell
.venv\Scripts\python scripts\benchmark.py "London" "Rome" --rounds 3
```

This runs the same trip with the agents in sequence and through the parallel
graph, clearing the HTTP cache between runs, and prints median timings. Quote
these measurements rather than estimates.

## Quick start

No API keys are needed for attractions, geocoding, itineraries, road routes, or
hotel listings.

```powershell
cd TravelPlanner
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run streamlit_app.py
```

Then open http://localhost:8501.

To run the REST API instead:

```powershell
.venv\Scripts\python main.py
```

- Interactive docs: http://127.0.0.1:8000/docs
- Capability check: http://127.0.0.1:8000/api/health

```powershell
.venv\Scripts\python scripts\test_pipeline.py "London" "Rome"
.venv\Scripts\python scripts\test_places.py "Kyoto, Japan" "history,culture"
.venv\Scripts\python scripts\test_hitl.py              # draft -> revise -> approve
.venv\Scripts\python scripts\test_mcp.py               # MCP server over stdio
.venv\Scripts\python scripts\test_supervisor_react.py  # ReAct loop, scripted model
.venv\Scripts\python scripts\test_intake.py [url-or-file ...]
```

## Deploying to Google Cloud Run

The `Dockerfile` runs the Streamlit app on `$PORT`. `.gcloudignore` and
`.dockerignore` keep `.env` and `.venv` out of the upload and the image; keys
reach the container through Secret Manager.

Cloud Run's always-free tier (per month, in us-central1, us-east1 or us-west1)
covers 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests, which
is about 50 active hours at 1 vCPU and 2 GiB. Set a budget alert before
deploying.

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Paste the value, then Enter, Ctrl+Z, Enter
gcloud secrets create openai-key --data-file=-
gcloud secrets create duffel-token --data-file=-

gcloud run deploy travelplanner --source . --region us-central1 `
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 `
  --session-affinity --timeout 900 --allow-unauthenticated `
  --set-secrets OPENAI_API_KEY=openai-key:latest,DUFFEL_ACCESS_TOKEN=duffel-token:latest `
  --set-env-vars CONTACT_URL=https://github.com/YOUR_USERNAME/TravelPlanner
```

The first deploy asks to enable the Cloud Run, Cloud Build and Artifact
Registry APIs. The runtime service account also needs the **Secret Manager
Secret Accessor** role on both secrets.

| Flag | Why |
|------|-----|
| `--memory 2Gi` | OCR, OpenCV and LangGraph exceed the 512 MiB default |
| `--max-instances 1` | Caps cost, and keeps every viewer on the instance that holds the in-memory checkpoints |
| `--session-affinity` | Keeps each Streamlit session on the same instance |
| `--timeout 900` | A cold first draft takes one to two minutes |
| `--min-instances 0` | Scales to zero when idle, so idle time is free |

Caveats:

- The URL is public, so anyone can spend your OpenAI quota. Set a usage limit
  in the OpenAI dashboard.
- The image is about 1.5–2 GB, more than Artifact Registry's 0.5 GB free
  storage, so expect a few cents a month. Delete superseded images.
- Paused plans live in memory and are lost when the instance scales to zero.
- YouTube often blocks requests from cloud IP ranges, so caption and video
  intake can fail there; PDFs, images and pasted text are unaffected.

To test the container locally first:

```powershell
docker build -t travelplanner .
docker run --rm -p 8080:8080 --env-file .env travelplanner
```

## How attraction ranking works

A plain geographic query returns every mapped bench and plaque, so the system
needs a relevance signal. Wikidata records how many language editions of
Wikipedia cover each subject, and that sitelink count is a strong proxy for
notability: the Colosseum has 147, a minor triumphal arch has 43. Places are
scored on that count, then modulated by category weight and the traveller's
stated interests.

The category itself cannot drive ranking. Wikidata types the Colosseum as an
amphitheatre and the Arch of Titus as a monument, so sorting by category would
put the arch first.

## How the itinerary is built

Two separate problems, solved in order:

1. **Which places share a day.** Attractions are clustered by coordinate with
   k-means (k = number of days), so a day stays within one part of the city.
   Clusters are then rebalanced against a per-day capacity, since k-means
   optimises compactness and knows nothing about how many stops fit in a day.
2. **What order to visit them.** Each day is a small travelling-salesman
   problem, solved with a nearest-neighbour heuristic and improved by 2-opt
   segment reversal. The day starts at its most notable place.

Slots are then filled against an hourly budget set by the chosen pace, with
walking or transit time inserted between consecutive stops and a lunch break
around midday.

## Configuration

Everything is optional. Copy the template and fill in what you have:

```powershell
copy .env.example .env
```

| Variable | Effect if unset |
|----------|-----------------|
| `DUFFEL_ACCESS_TOKEN` | No flight offers; road routes still work. A free `duffel_test_` token from the [Duffel dashboard](https://app.duffel.com/join) returns test-mode offers |
| `OPENAI_API_KEY` | Supervisor uses rules instead of ReAct; intake skips Whisper and vision and extracts heuristically; itinerary notes come from Wikipedia |
| `CONTACT_URL` | Defaults to a placeholder repo URL |

`CONTACT_URL` matters more than it looks. Wikimedia rejects requests without
identifying contact info in the User-Agent, and Nominatim additionally blocks
placeholder domains such as `example.com`. Point it at your own repo or e-mail.

## Project layout

```
TravelPlanner/
├── Dockerfile                  Container for Cloud Run
├── main.py                     FastAPI entry point
├── streamlit_app.py            Web UI
├── app/
│   ├── config.py               Settings and capability flags
│   ├── agents/
│   │   ├── base.py             Timing, error handling, skip semantics
│   │   ├── orchestrator.py     Supervisor graph, checkpointer, interrupt
│   │   ├── supervisor.py       ReAct loop over MCP tools + field-impact rules
│   │   └── *_agent.py          Destination, itinerary, transport, hotel, weather, booking
│   ├── mcp/
│   │   ├── server.py           MCP server exposing the providers as tools
│   │   └── client.py           Stdio client, OpenAI tool schemas
│   ├── intake/
│   │   ├── agent.py            Combine evidence, extract the shared schema
│   │   ├── sources.py          Captions, yt-dlp, frames, PDF text, OCR
│   │   └── models.py           Whisper, GPT-4o vision, structured extraction
│   ├── providers/
│   │   ├── http.py             Caching, retries, per-host rate limiting
│   │   ├── geocoding.py        Nominatim
│   │   ├── places.py           Wikidata + Wikipedia
│   │   ├── flights.py          Duffel flight offers, nearest-airport lookup
│   │   ├── hotels.py           OpenStreetMap accommodation listings
│   │   ├── routing.py          OSRM road routes, haversine fallback
│   │   ├── weather.py          Open-Meteo forecast and archive
│   │   └── currency.py         ECB rates for converting fares to USD
│   ├── models/travel.py        Plan, review and booking models
│   ├── models/intake.py        Shared TripIntake schema
│   └── services/llm_service.py Optional itinerary narration
└── scripts/                    Live-data test harnesses
```

## Design notes

**Prices are never invented.** Every price field is optional, and a missing
price is reported as missing. The cost estimate totals only components that
carry a real quote and warns when it is partial.

**Failure is per-agent.** `BaseAgent.execute` distinguishes a genuine failure
from a skip caused by an unavailable prerequisite. One provider going down
degrades that section of the plan and surfaces a warning, rather than losing the
whole trip.

**Public endpoints are treated as shared infrastructure.** The HTTP layer
enforces a minimum interval per host, caches responses for an hour, and retries
throttling and gateway errors with backoff. Nominatim's policy is one request
per second and Overpass instances throttle aggressively.

**Fallback chains are explicit.** Places fall back from Wikidata SPARQL to
Wikipedia geosearch; routing from OSRM to a straight-line estimate that is flagged as an
estimate so the transport agent knows not to offer a drive across an ocean.

## Booking stops at the hand-off

The booking agent never takes payment. Real booking needs payment processing,
inventory holds, and idempotency guarantees against double-charging, none of
which can be demonstrated honestly against a test API. After approval it
builds links prefilled with the approved route, dates and party size: Google
Flights or Maps for transport, the hotel's site or a Booking.com search for
stays, and official sites for ticketed attractions.

## Known limitations

- Attraction coverage depends on Wikidata, so small towns return few results.
  Querying a dense city can take 10–30 seconds on a cold cache.
- Duffel test mode returns test offers (including the fictional "Duffel
  Airways"), not real fares. Prices are labelled `Duffel (test mode)`; use a
  live token for real inventory. Fares quoted in other currencies are converted
  to USD at the day's ECB rate, and the original amount is kept alongside.
- There is no free source of priced hotel inventory since Amadeus closed its
  self-service API in July 2026, so stays show real properties without rates and
  the booking agent links to a prefilled live search.
- Intra-city travel times are speed-based estimates from real coordinates, not
  routed transit directions.
- `TTLCache` and the LangGraph checkpointer are per-process and in-memory; a
  multi-worker deployment would want Redis and a database-backed saver.
- Instagram and TikTok often require a signed-in session for downloads; when
  yt-dlp is refused, intake reports it and carries on with the other sources.
- The no-key intake heuristic is crude. Review the extracted places before
  planning.
