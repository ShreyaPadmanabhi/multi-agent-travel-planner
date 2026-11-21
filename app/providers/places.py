"""Real attraction discovery from Wikidata, enriched with Wikipedia extracts.

Wikidata is the primary source because it carries coordinates, a type hierarchy,
and — critically — a sitelink count that acts as a notability ranking. Without
some popularity signal, a geo query returns every mapped bench and plaque rather
than the places a traveller actually wants. Wikipedia geosearch is the fallback
when the SPARQL endpoint is unavailable.
"""

from __future__ import annotations

import re
import urllib.parse

from app.config import settings
from app.models.travel import GeoLocation, Place
from app.providers.http import ProviderError, request_json

# Root classes whose subclasses describe things worth visiting. Matching on
# `P279*` from these roots keeps recall high without enumerating hundreds of
# specific types (a palace, basilica and observation tower all resolve here).
ALLOWED_ROOTS = [
    "Q811979",   # architectural structure
    "Q33506",    # museum
    "Q570116",   # tourist attraction
    "Q22698",    # park
    "Q4989906",  # monument
    "Q839954",   # archaeological site
    "Q174782",   # public square
    "Q271669",   # landform
    "Q1107656",  # garden
    "Q57821",    # fortification
]

# Types that pass the allowlist (they are structures) but are not destinations.
# Kept deliberately narrow: broad classes like "organization" would also strike
# out museums, and "stadium" would strike out the Colosseum. Anything merely
# low-value is damped by CATEGORY_WEIGHT instead of excluded here.
EXCLUDED_TYPES = {
    "Q1248784",   # airport
    "Q644371",    # international airport
    "Q55488",     # railway station
    "Q928830",    # metro station
    "Q548662",    # public transport stop
    "Q3918",      # university
    "Q3914",      # school
    "Q9826",      # high school
    "Q16917",     # hospital
    "Q27686",     # hotel
    "Q11707",     # restaurant
    "Q327333",    # government agency
    "Q1021645",   # office building
    "Q699405",    # shopping centre
}

# Map Wikidata types onto the interest categories the UI exposes. Checked in
# order, so more specific types must come first.
TYPE_CATEGORIES: list[tuple[set[str], str]] = [
    ({"Q33506", "Q207694", "Q588140", "Q2772772", "Q1007870", "Q3329412"}, "museum"),
    ({"Q16970", "Q120560", "Q2977", "Q317557", "Q44613", "Q34627", "Q32815", "Q845945", "Q56242215", "Q1128397"}, "religious"),
    ({"Q23413", "Q751876", "Q16560", "Q57821", "Q1785071", "Q82117", "Q1620897"}, "historic"),
    ({"Q839954", "Q4989906", "Q179700", "Q5003624", "Q1043849"}, "monument"),
    ({"Q22698", "Q1107656", "Q22746", "Q194195", "Q43501"}, "park"),
    ({"Q1440300", "Q200334", "Q11303", "Q1440476"}, "viewpoint"),
    ({"Q40080", "Q23397", "Q8502", "Q4022", "Q271669"}, "nature"),
    ({"Q24354", "Q153562", "Q41253"}, "entertainment"),
    ({"Q330284", "Q11315"}, "food"),
    ({"Q174782", "Q54114", "Q79007", "Q12280"}, "landmark"),
    ({"Q483110", "Q1154710"}, "sports"),
]

# Multiplies the notability score. Sports venues and transport-adjacent sites
# are real places but rarely the reason someone visits a city, so they are
# damped rather than excluded.
CATEGORY_WEIGHT = {
    "museum": 1.0,
    "historic": 1.0,
    "monument": 0.9,
    "religious": 0.9,
    "landmark": 1.0,
    "viewpoint": 1.0,
    "park": 0.85,
    "nature": 0.85,
    "food": 0.9,
    "entertainment": 0.7,
    "attraction": 1.0,
    "sports": 0.25,
}

# Rough visit durations in hours, used by the itinerary scheduler. Major sites
# get a longer slot because queueing is usually the bulk of the visit.
VISIT_HOURS = {
    "museum": 2.5,
    "historic": 2.0,
    "monument": 1.25,
    "religious": 1.0,
    "landmark": 1.0,
    "viewpoint": 1.5,
    "park": 1.5,
    "nature": 2.0,
    "food": 1.5,
    "entertainment": 2.0,
    "sports": 2.0,
    "attraction": 1.5,
}

INTEREST_CATEGORIES = {
    "culture": {"museum", "religious", "landmark", "entertainment"},
    "history": {"historic", "monument", "museum", "religious"},
    "art": {"museum", "entertainment"},
    "food": {"food"},
    "nature": {"park", "nature", "viewpoint"},
    "adventure": {"nature", "viewpoint", "sports"},
    "architecture": {"landmark", "religious", "historic", "viewpoint"},
    "nightlife": {"entertainment", "food"},
}

SPARQL_TEMPLATE = """
SELECT ?item ?itemLabel ?coord ?sitelinks ?article
       (GROUP_CONCAT(DISTINCT STRAFTER(STR(?type), "entity/"); separator=",") AS ?types)
       (SAMPLE(?image) AS ?img)
       (SAMPLE(?site) AS ?website)
WHERE {
  SERVICE wikibase:around {
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point(%(lon)f %(lat)f)"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "%(radius)d" .
  }
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= %(min_sitelinks)d)
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
  ?item wdt:P31 ?type .
  ?type wdt:P279* ?root .
  VALUES ?root { %(roots)s }
  OPTIONAL { ?item wdt:P18 ?image . }
  OPTIONAL { ?item wdt:P856 ?site . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?item ?itemLabel ?coord ?sitelinks ?article
ORDER BY DESC(?sitelinks)
LIMIT %(limit)d
"""

POINT_PATTERN = re.compile(r"Point\(([-\d.]+)\s+([-\d.]+)\)")


class PlacesProvider:
    source = "Wikidata + Wikipedia"

    async def find_places(
        self,
        location: GeoLocation,
        interests: list[str],
        limit: int = 30,
    ) -> list[Place]:
        try:
            places = await self._query_wikidata(location)
        except ProviderError:
            places = await self._query_wikipedia_geosearch(location)

        if not places:
            raise ProviderError(
                f"No notable attractions found near {location.display_name}. "
                "Try a nearby larger city."
            )

        places = self._rank(places, interests, location)[:limit]
        await self._enrich_with_wikipedia(places)
        return places

    async def _query_wikidata(self, location: GeoLocation) -> list[Place]:
        query = SPARQL_TEMPLATE % {
            "lat": location.latitude,
            "lon": location.longitude,
            "radius": settings.place_search_radius_km,
            "min_sitelinks": settings.place_min_sitelinks,
            "roots": " ".join(f"wd:{qid}" for qid in ALLOWED_ROOTS),
            "limit": settings.place_candidate_limit,
        }

        payload = await request_json(
            "GET",
            settings.wikidata_sparql_url,
            params={"query": query, "format": "json"},
            limiter="wikidata",
            retries=1,
        )

        places: list[Place] = []
        seen: set[str] = set()

        for row in payload.get("results", {}).get("bindings", []):
            name = row.get("itemLabel", {}).get("value", "")
            # An unresolved label falls back to the raw QID; skip those.
            if not name or re.fullmatch(r"Q\d+", name):
                continue

            key = name.lower()
            if key in seen:
                continue

            coords = POINT_PATTERN.match(row.get("coord", {}).get("value", ""))
            if not coords:
                continue
            longitude, latitude = float(coords.group(1)), float(coords.group(2))

            types = set(row.get("types", {}).get("value", "").split(","))
            if types & EXCLUDED_TYPES:
                continue

            category = self._categorise(types)
            article = row.get("article", {}).get("value", "")
            title = urllib.parse.unquote(article.rsplit("/", 1)[-1]).replace("_", " ")

            seen.add(key)
            places.append(
                Place(
                    name=name,
                    category=category,
                    estimated_hours=VISIT_HOURS.get(category, 1.5),
                    latitude=latitude,
                    longitude=longitude,
                    wikipedia_title=title,
                    image_url=row.get("img", {}).get("value", ""),
                    website=row.get("website", {}).get("value", ""),
                    popularity=int(row.get("sitelinks", {}).get("value", 0)),
                    source=self.source,
                )
            )

        return places

    async def _query_wikipedia_geosearch(self, location: GeoLocation) -> list[Place]:
        """Fallback when the SPARQL endpoint is down. Fast, but unfiltered by type."""
        payload = await request_json(
            "GET",
            settings.wikipedia_api_url,
            params={
                "action": "query",
                "format": "json",
                "list": "geosearch",
                "gscoord": f"{location.latitude}|{location.longitude}",
                "gsradius": min(settings.place_search_radius_km * 1000, 10000),
                "gslimit": 100,
            },
            limiter="wikipedia",
        )

        places: list[Place] = []
        for entry in payload.get("query", {}).get("geosearch", []):
            title = entry.get("title", "")
            if not title or ":" in title:
                continue
            places.append(
                Place(
                    name=title,
                    category="attraction",
                    estimated_hours=1.5,
                    latitude=float(entry["lat"]),
                    longitude=float(entry["lon"]),
                    wikipedia_title=title,
                    popularity=0,
                    source="Wikipedia geosearch",
                )
            )
        return places

    @staticmethod
    def _categorise(types: set[str]) -> str:
        for type_ids, category in TYPE_CATEGORIES:
            if types & type_ids:
                return category
        return "attraction"

    def _rank(self, places: list[Place], interests: list[str], location: GeoLocation) -> list[Place]:
        """Score by notability, then modulate by category and stated interests.

        Notability has to dominate: category alone would rank a minor triumphal
        arch above the Colosseum simply because of how Wikidata types them.
        """
        preferred: set[str] = set()
        for interest in interests:
            preferred |= INTEREST_CATEGORIES.get(interest.lower(), set())

        destination_name = location.name.lower()

        def score(place: Place) -> float:
            if place.name.lower() == destination_name:
                return -1.0  # The city's own article is not an attraction.
            value = float(place.popularity or 1)
            value *= CATEGORY_WEIGHT.get(place.category, 0.9)
            if preferred and place.category in preferred:
                value *= 1.6
            return value

        return sorted(places, key=score, reverse=True)

    async def _enrich_with_wikipedia(self, places: list[Place]) -> None:
        """Attach real encyclopaedic summaries, batching up to 20 titles per call."""
        titled = [p for p in places if p.wikipedia_title]
        if not titled:
            return

        for start in range(0, len(titled), 20):
            batch = titled[start : start + 20]
            try:
                payload = await request_json(
                    "GET",
                    settings.wikipedia_api_url,
                    params={
                        "action": "query",
                        "format": "json",
                        "prop": "extracts|pageimages",
                        "exintro": 1,
                        "explaintext": 1,
                        "redirects": 1,
                        "piprop": "thumbnail",
                        "pithumbsize": 480,
                        "titles": "|".join(p.wikipedia_title for p in batch),
                    },
                    limiter="wikipedia",
                )
            except ProviderError:
                continue  # Summaries are enrichment, never fatal.

            query = payload.get("query", {})
            extracts: dict[str, str] = {}
            thumbs: dict[str, str] = {}
            for page in query.get("pages", {}).values():
                title = page.get("title", "").lower()
                if extract := page.get("extract"):
                    extracts[title] = extract
                if thumb := (page.get("thumbnail") or {}).get("source"):
                    thumbs[title] = thumb

            # Titles may be rewritten by normalisation or redirects.
            aliases: dict[str, str] = {}
            for kind in ("normalized", "redirects"):
                for item in query.get(kind, []):
                    aliases[item["from"].lower()] = item["to"].lower()

            for place in batch:
                key = place.wikipedia_title.lower()
                for _ in range(3):
                    if key in extracts or key not in aliases:
                        break
                    key = aliases[key]

                if summary := extracts.get(key):
                    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
                    place.description = " ".join(sentences[:2])[:400]
                if not place.image_url and (thumb := thumbs.get(key)):
                    place.image_url = thumb
