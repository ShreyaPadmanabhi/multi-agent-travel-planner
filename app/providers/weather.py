"""Daily weather from Open-Meteo (keyless).

Open-Meteo forecasts about 16 days ahead. For trips further out there is no
honest forecast, so the provider reports what the same dates looked like last
year and labels it as such.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.config import settings
from app.models.travel import DayWeather, GeoLocation
from app.providers.http import ProviderError, request_json

FORECAST_HORIZON_DAYS = 15

# WMO weather interpretation codes, grouped into plain-language summaries.
WMO_SUMMARIES = [
    ({0}, "Clear sky"),
    ({1, 2}, "Partly cloudy"),
    ({3}, "Overcast"),
    ({45, 48}, "Fog"),
    ({51, 53, 55, 56, 57}, "Drizzle"),
    ({61, 63, 65, 66, 67, 80, 81, 82}, "Rain"),
    ({71, 73, 75, 77, 85, 86}, "Snow"),
    ({95, 96, 99}, "Thunderstorms"),
]


def describe_weather_code(code: int | None) -> str:
    if code is None:
        return "Unknown"
    for codes, label in WMO_SUMMARIES:
        if code in codes:
            return label
    return "Mixed"


def _shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:  # 29 February
        return value.replace(year=value.year + years, day=28)


class WeatherProvider:
    source = "Open-Meteo"

    async def daily(self, location: GeoLocation, start: date, end: date) -> list[DayWeather]:
        horizon = date.today() + timedelta(days=FORECAST_HORIZON_DAYS)
        if date.today() <= start and end <= horizon:
            return await self._query(settings.open_meteo_url, location, start, end, "forecast", 0)
        # Beyond the forecast window, or in the past: report last year's observations.
        return await self._query(
            settings.open_meteo_archive_url, location, start, end, "same dates last year", -1
        )

    async def _query(
        self,
        url: str,
        location: GeoLocation,
        start: date,
        end: date,
        basis: str,
        year_shift: int,
    ) -> list[DayWeather]:
        query_start, query_end = _shift_year(start, year_shift), _shift_year(end, year_shift)
        daily_fields = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        if basis == "forecast":
            daily_fields.append("precipitation_probability_max")

        payload = await request_json(
            "GET",
            url,
            params={
                "latitude": round(location.latitude, 4),
                "longitude": round(location.longitude, 4),
                "daily": ",".join(daily_fields),
                "timezone": "auto",
                "start_date": query_start.isoformat(),
                "end_date": query_end.isoformat(),
            },
            limiter="open-meteo",
        )
        daily = payload.get("daily") or {}
        days = daily.get("time") or []
        if not days:
            raise ProviderError("Open-Meteo returned no daily weather.")

        def column(name: str) -> list:
            return daily.get(name) or [None] * len(days)

        codes, highs, lows = column("weather_code"), column("temperature_2m_max"), column("temperature_2m_min")
        rain, chance = column("precipitation_sum"), column("precipitation_probability_max")

        return [
            DayWeather(
                date=_shift_year(date.fromisoformat(day), -year_shift),
                summary=describe_weather_code(codes[i]),
                temp_max_c=highs[i],
                temp_min_c=lows[i],
                precipitation_mm=rain[i],
                precipitation_probability=chance[i],
                basis=basis,
            )
            for i, day in enumerate(days)
        ]
