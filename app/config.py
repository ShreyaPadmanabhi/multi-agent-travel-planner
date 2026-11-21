from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Strip whitespace: `docker run --env-file` and pasted secrets keep stray
    # spaces that the .env loader would otherwise have removed.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", str_strip_whitespace=True
    )

    # Optional: LLM-driven itinerary narration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Supervisor ReAct loop, vision over frames/pamphlets, and speech-to-text.
    openai_supervisor_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o"
    openai_transcription_model: str = "whisper-1"
    supervisor_max_steps: int = 6

    # Optional: Duffel flight offers. A duffel_test_ token is free and returns
    # test-mode offers; a live token returns real airline inventory.
    duffel_access_token: str = ""
    duffel_base_url: str = "https://api.duffel.com"
    frankfurter_url: str = "https://api.frankfurter.dev/v1/latest"

    # Keyless open-data endpoints
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    wikidata_sparql_url: str = "https://query.wikidata.org/sparql"
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    osrm_url: str = "https://router.project-osrm.org"
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"

    # Wikimedia and Nominatim both require an identifying User-Agent with real
    # contact info. Wikimedia rejects bare tokens and browser strings; Nominatim
    # additionally blocks placeholder domains such as example.com.
    app_name: str = "TravelPlanner"
    app_version: str = "2.0"
    contact_url: str = "https://github.com/travelplanner"

    http_timeout_seconds: float = 90.0
    cache_ttl_seconds: int = 3600

    # Search tuning
    place_search_radius_km: int = 15
    place_min_sitelinks: int = 6
    place_candidate_limit: int = 80
    hotel_search_radius_km: int = 12
    must_visit_radius_km: int = 60

    # Multimodal intake limits. Whisper's API rejects files over 25 MB.
    intake_max_media_mb: int = 25
    intake_video_frames: int = 6
    intake_max_pdf_pages: int = 30

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def user_agent(self) -> str:
        return f"{self.app_name}/{self.app_version} ({self.contact_url})"

    @property
    def flights_enabled(self) -> bool:
        return bool(self.duffel_access_token)

    @property
    def flight_mode(self) -> str:
        if not self.duffel_access_token:
            return "off"
        return "test" if self.duffel_access_token.startswith("duffel_test_") else "live"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


settings = Settings()
