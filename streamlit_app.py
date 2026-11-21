import asyncio
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from app.agents.orchestrator import TravelOrchestrator
from app.config import settings
from app.intake import IntakeAgent
from app.models.travel import TravelPlan, TravelPlanRequest

st.set_page_config(page_title="TravelPlanner", page_icon="✈️", layout="wide")

STATUS_ICONS = {"completed": "✅", "skipped": "⚠️", "failed": "❌"}
FLIGHT_LABELS = {"off": "off (add DUFFEL_ACCESS_TOKEN)", "test": "Duffel test mode", "live": "Duffel live"}
INTERESTS = ["history", "culture", "art", "food", "nature", "adventure", "architecture", "nightlife"]

FORM_DEFAULTS = {
    "origin": "London",
    "destination": "Rome",
    "start_date": date.today() + timedelta(days=45),
    "end_date": date.today() + timedelta(days=48),
    "travelers": 2,
    "budget": 2500,
    "pace": "balanced",
    "travel_mode": "any",
    "interests": ["history", "culture"],
    "must_visit": "",
    "preferred_hotels": "",
}
for key, value in FORM_DEFAULTS.items():
    st.session_state.setdefault(key, value)
# Widget values can only be set before the widgets render, so intake results
# are staged here and applied at the top of the next rerun.
st.session_state.update(st.session_state.pop("prefill", {}))


@st.cache_resource
def get_orchestrator() -> TravelOrchestrator:
    # Cached across reruns so paused plans stay in the checkpointer.
    return TravelOrchestrator()


def run(coro):
    return asyncio.run(coro)


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.replace(",", "\n").splitlines() if line.strip()]


st.title("✈️ TravelPlanner")
st.caption(
    "A LangGraph supervisor runs four specialist agents in parallel on live open data, "
    "pauses for your approval, then prepares booking links. Trip material — vlogs, reels, "
    "PDFs, pamphlets — can be imported to seed the plan."
)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Plan a trip")
    st.text_input("From", key="origin")
    st.text_input("To", key="destination")
    col1, col2 = st.columns(2)
    col1.date_input("Start", key="start_date")
    col2.date_input("End", key="end_date")
    st.number_input("Travellers", min_value=1, max_value=9, key="travelers")
    st.number_input("Budget (USD, 0 = no limit)", min_value=0, step=250, key="budget")
    st.select_slider("Pace", options=["relaxed", "balanced", "packed"], key="pace")
    st.selectbox("Transport", ["any", "flight", "ground"], key="travel_mode")
    st.multiselect("Interests", INTERESTS, key="interests")
    st.text_area("Must-visit places (one per line)", key="must_visit", height=90)
    st.text_area("Preferred hotels (one per line)", key="preferred_hotels", height=68)

    submitted = st.button("Draft my trip", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Capabilities")
    st.markdown(
        f"- Supervisor: **{'ReAct over MCP tools' if settings.llm_enabled else 'rules (add OPENAI_API_KEY for ReAct)'}**\n"
        f"- Flight offers: **{FLIGHT_LABELS[settings.flight_mode]}**\n"
        f"- Whisper + GPT-4o vision intake: **{'on' if settings.llm_enabled else 'off'}**\n"
        "- Captions, PDF text and OCR intake: **on**"
    )

# ------------------------------------------------------------------- intake

with st.expander("📥 Import trip material (YouTube, reels, PDFs, pamphlets)", expanded="thread_id" not in st.session_state):
    urls_text = st.text_area("Video links, one per line", placeholder="https://www.youtube.com/watch?v=...")
    uploads = st.file_uploader(
        "Files",
        type=["pdf", "png", "jpg", "jpeg", "webp", "mp4", "mov", "m4a", "mp3", "txt"],
        accept_multiple_files=True,
    )
    pasted = st.text_area("Or paste an itinerary")
    hint = st.text_input("Destination (helps when the material does not say)", value="")

    if st.button("Extract trip"):
        with tempfile.TemporaryDirectory(prefix="upload-") as tmp:
            files = []
            for index, upload in enumerate(uploads or []):
                path = Path(tmp) / f"upload-{index}{Path(upload.name).suffix.lower()}"
                path.write_bytes(upload.getvalue())
                files.append((path, upload.name))
            with st.spinner("Reading captions, transcribing, running OCR and vision..."):
                try:
                    result = run(
                        IntakeAgent().process(
                            urls=split_lines(urls_text), files=files, text=pasted, destination_hint=hint
                        )
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    result = None
        if result:
            st.session_state["intake"] = result
            trip = result.trip
            if trip.destination:
                # Keep the form's own values for anything the material does not state.
                request = result.to_request(
                    origin=st.session_state["origin"],
                    start_date=trip.start_date or st.session_state["start_date"],
                    travelers=trip.travelers or st.session_state["travelers"],
                )
                st.session_state["prefill"] = dict(
                    destination=request.destination,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    travelers=request.travelers,
                    pace=request.pace,
                    interests=[i for i in request.interests if i in INTERESTS] or st.session_state["interests"],
                    must_visit="\n".join(request.must_visit),
                    preferred_hotels="\n".join(request.preferred_hotels),
                )
            st.rerun()

    if result := st.session_state.get("intake"):
        if result.trip.destination:
            st.success(
                f"Extracted with **{result.extraction}** method from {len(result.sources)} source(s). "
                "The trip form has been filled in — review it, then draft the trip."
            )
        else:
            # Pinning these places to whatever city is in the form would be wrong.
            st.info(
                f"Extracted {len(result.trip.places)} place(s), but the material does not name a destination. "
                "Enter it in **Destination** above and extract again to fill the trip form."
            )
        for source in result.sources:
            st.caption(f"{source.kind} · {source.reference} · {', '.join(source.methods) or 'nothing read'} · {source.characters} chars")
        for warning in result.warnings:
            st.warning(warning, icon="⚠️")
        if result.trip.places:
            st.dataframe(
                pd.DataFrame([p.model_dump() for p in result.trip.places]), hide_index=True, use_container_width=True
            )

# --------------------------------------------------------------- start plan

if submitted:
    if st.session_state["end_date"] < st.session_state["start_date"]:
        st.error("The end date must be on or after the start date.")
        st.stop()
    request = TravelPlanRequest(
        origin=st.session_state["origin"],
        destination=st.session_state["destination"],
        start_date=st.session_state["start_date"],
        end_date=st.session_state["end_date"],
        travelers=int(st.session_state["travelers"]),
        budget_usd=float(st.session_state["budget"]) if st.session_state["budget"] > 0 else None,
        interests=st.session_state["interests"],
        travel_mode=st.session_state["travel_mode"],
        pace=st.session_state["pace"],
        must_visit=split_lines(st.session_state["must_visit"]),
        preferred_hotels=split_lines(st.session_state["preferred_hotels"]),
    )
    with st.spinner("Supervisor is dispatching the specialists..."):
        thread_id, plan = run(get_orchestrator().start(request))
    st.session_state["thread_id"] = thread_id
    st.session_state["plan"] = plan

plan: TravelPlan | None = st.session_state.get("plan")
if plan is None:
    st.info("Set trip details in the sidebar (or import material above), then select **Draft my trip**.")
    st.subheader("How it works")
    st.markdown(
        """
        | Stage | Agent | Responsibility |
        |------|-------|----------------|
        | Intake | **IntakeAgent** | Captions / Whisper, OCR and GPT-4o vision turn vlogs, reels, PDFs and pamphlets into one trip schema |
        | 1 | **DestinationAgent** | Geocodes the trip, finds notable attractions, pins the places you asked for |
        | 2 | **Supervisor** | ReAct loop over MCP tools decides what to run; on revisions re-runs only what changed |
        | 3 (parallel) | **Itinerary · Transport · Hotel · Weather** | Day routes, flights and roads, stays, daily weather |
        | 4 | **You** | Approve, ask for changes, or reject — the graph pauses until you decide |
        | 5 | **BookingAgent** | Prefilled booking links for the approved plan; no payment taken |
        """
    )
    st.stop()

thread_id = st.session_state["thread_id"]

# ------------------------------------------------------------------- review

status_label = {
    "awaiting_approval": "🟡 Awaiting your approval",
    "approved": "🟢 Approved — booking links ready",
    "rejected": "🔴 Rejected",
    "failed": "❌ Planning failed",
}.get(plan.status, plan.status)
st.subheader(f"{plan.destination_label.split(',')[0] or plan.request.destination} · {status_label}")

if plan.status == "awaiting_approval":
    with st.container(border=True):
        st.markdown("**Review this draft**")
        feedback = st.text_input("Feedback", placeholder="e.g. too packed — slow it down and add Villa Borghese")
        c1, c2, c3 = st.columns(3)
        new_pace = c1.selectbox("Pace", ["(keep)", "relaxed", "balanced", "packed"])
        new_mode = c2.selectbox("Transport", ["(keep)", "any", "flight", "ground"])
        add_stop = c3.text_input("Add a must-visit")

        changes: dict = {}
        if new_pace != "(keep)":
            changes["pace"] = new_pace
        if new_mode != "(keep)":
            changes["travel_mode"] = new_mode
        if add_stop.strip():
            changes["must_visit"] = plan.request.must_visit + [add_stop.strip()]

        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Approve & prepare bookings", type="primary", use_container_width=True):
            st.session_state["plan"] = run(get_orchestrator().review(thread_id, "approve"))
            st.rerun()
        if b2.button("🔁 Request changes", use_container_width=True, disabled=not (feedback or changes)):
            with st.spinner("Supervisor is re-running the affected specialists..."):
                st.session_state["plan"] = run(get_orchestrator().review(thread_id, "revise", feedback, changes))
            st.rerun()
        if b3.button("🗑️ Reject", use_container_width=True):
            st.session_state["plan"] = run(get_orchestrator().review(thread_id, "reject", feedback))
            st.rerun()
elif st.button("Start a new plan"):
    for key in ("plan", "thread_id"):
        st.session_state.pop(key, None)
    st.rerun()

metrics = st.columns(5)
metrics[0].metric("Places found", len(plan.places))
metrics[1].metric("Days scheduled", len(plan.itinerary))
metrics[2].metric("Transport options", len(plan.travel_options))
metrics[3].metric("Priced total", f"${plan.estimated_cost_usd:,.0f}" if plan.estimated_cost_usd else "n/a")
metrics[4].metric("Revisions", sum(r.action == "revise" for r in plan.review_history))

for warning in plan.warnings:
    st.warning(warning, icon="⚠️")

with st.expander("Supervisor & agent activity", expanded=True):
    if plan.supervisor:
        decision = plan.supervisor
        st.markdown(
            f"**Supervisor ({decision.mode})** dispatched: {', '.join(decision.specialists) or 'nothing'}  \n"
            f"{decision.rationale}"
        )
        if decision.request_changes:
            st.caption(f"Request changes applied: {decision.request_changes}")
    for step in plan.supervisor_trace:
        st.markdown(
            f"- 💭 *{step.thought or '—'}*  \n  🔧 `{step.action}({step.action_input})`"
            + (f"  \n  👁️ <span style='color:#64748b'>{step.observation[:220]}</span>" if step.observation else ""),
            unsafe_allow_html=True,
        )
    for result in plan.agent_results:
        icon = STATUS_ICONS.get(result.status.value, "•")
        st.markdown(
            f"{icon} **{result.agent_name}** — {result.message}  \n"
            f"<span style='color:#64748b;font-size:0.85em'>{result.duration_ms} ms · {result.source}</span>",
            unsafe_allow_html=True,
        )

tabs = st.tabs(["📅 Itinerary", "📍 Places", "🗺️ Map", "🚆 Transport", "🏨 Stays", "🌦️ Weather", "🎟️ Bookings"])
tab_itinerary, tab_places, tab_map, tab_transport, tab_hotels, tab_weather, tab_bookings = tabs

with tab_itinerary:
    if not plan.itinerary:
        st.info("No itinerary could be built.")
    weather_by_date = {w.date: w for w in plan.weather}
    for day in plan.itinerary:
        st.subheader(f"Day {day.day} — {day.date:%a %d %b}")
        forecast = weather_by_date.get(day.date)
        weather_note = ""
        if forecast and forecast.temp_max_c is not None:
            weather_note = f" · {forecast.summary}, {forecast.temp_max_c:.0f}°C"
        st.caption(f"{day.theme} · about {day.walking_distance_km} km on foot{weather_note}")
        for item in day.items:
            leg = ""
            if item.travel_minutes_from_previous:
                leg = f" &nbsp;·&nbsp; <em>{item.travel_minutes_from_previous} min {item.travel_mode}</em>"
            pin = " 📌" if any(p.pinned and p.name == item.place_name for p in plan.places) else ""
            st.markdown(
                f"**{item.start_time} – {item.end_time}** &nbsp; {item.activity}{pin}{leg}",
                unsafe_allow_html=True,
            )
            if item.notes:
                st.caption(item.notes)
        st.divider()

with tab_places:
    for place in plan.places:
        label = f"{'📌 ' if place.pinned else ''}{place.name} · {place.category}"
        with st.expander(label):
            columns = st.columns([1, 2]) if place.image_url else [st.container()]
            if place.image_url:
                columns[0].image(place.image_url, use_container_width=True)
                body = columns[1]
            else:
                body = st
            body.write(place.description or "No description available.")
            body.caption(
                f"Suggested visit {place.estimated_hours}h · "
                f"cited by {place.popularity} Wikipedia editions · source: {place.source}"
            )
            if place.website:
                body.markdown(f"[Official site]({place.website})")

with tab_map:
    if plan.places:
        st.map(pd.DataFrame([{"lat": p.latitude, "lon": p.longitude} for p in plan.places]), zoom=11)
        st.caption("Real coordinates for every attraction.")
    else:
        st.info("No coordinates to plot.")

with tab_transport:
    if plan.recommended_travel:
        best = plan.recommended_travel
        price = f"${best.price_usd:,.2f}" if best.price_usd is not None else "no live price"
        st.info(f"**Recommended:** {best.provider} · {best.duration_label} · {price}")
    if not plan.travel_options:
        st.warning("No transport options were returned for this route.")
    for option in plan.travel_options:
        price = f"${option.price_usd:,.2f}" if option.price_usd is not None else "price unavailable"
        st.markdown(f"**{option.provider}** ({option.mode}) — {option.duration_label} — {price}")
        details = []
        if option.distance_km:
            details.append(f"{option.distance_km:,.0f} km")
        if option.stops is not None:
            details.append(f"{option.stops} stop(s)")
        if option.booking_class:
            details.append(option.booking_class)
        details.append(f"source: {option.source}")
        st.caption(" · ".join(details))
        for segment in option.segments:
            st.caption(
                f"    {segment.flight_number}: {segment.origin} "
                f"{segment.departure_time:%d %b %H:%M} → {segment.destination} "
                f"{segment.arrival_time:%d %b %H:%M}"
            )

with tab_hotels:
    if plan.recommended_hotel:
        best = plan.recommended_hotel
        rate = f"${best.price_per_night_usd:,.2f}/night" if best.price_per_night_usd else "no live rate"
        st.info(f"**Recommended:** {best.name} · {rate}")
    for hotel in plan.hotels:
        rate = (
            f"${hotel.price_per_night_usd:,.2f}/night (${hotel.total_price_usd:,.0f} total)"
            if hotel.price_per_night_usd
            else "rate unavailable"
        )
        st.markdown(f"**{hotel.name}** — {rate}")
        details = []
        if hotel.stars:
            details.append("⭐" * hotel.stars)
        if hotel.distance_to_center_km is not None:
            details.append(f"{hotel.distance_to_center_km} km from centre")
        if hotel.address:
            details.append(hotel.address)
        details.append(f"source: {hotel.source}")
        st.caption(" · ".join(details))

with tab_weather:
    if plan.weather:
        st.caption(f"Basis: {plan.weather[0].basis} · source: Open-Meteo")
        st.dataframe(
            pd.DataFrame([w.model_dump() for w in plan.weather]).drop(columns=["basis"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No weather data.")

with tab_bookings:
    if plan.status != "approved":
        st.info("Booking links are prepared once you approve the plan.")
    for booking in plan.bookings:
        price = f" · ${booking.price_usd:,.0f}" if booking.price_usd is not None else ""
        st.markdown(f"**{booking.title}** — {booking.provider}{price}  \n[{booking.url[:80]}]({booking.url})")
        if booking.notes:
            st.caption(booking.notes)
    if plan.bookings:
        st.caption("No payment is taken here; each link opens the provider prefilled with your dates.")

with st.expander("Trip summary & review history"):
    st.text(plan.summary)
    if plan.cost_breakdown:
        st.write("Cost breakdown (priced components only):")
        st.json(plan.cost_breakdown)
    for record in plan.review_history:
        st.caption(f"{record.reviewed_at:%H:%M:%S} · {record.action} · {record.feedback} {record.changes or ''}")
    st.caption(f"Data sources: {', '.join(plan.data_sources) or 'none'}")
