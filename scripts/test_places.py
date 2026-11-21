import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.providers.geocoding import GeocodingProvider
from app.providers.places import PlacesProvider


async def main(city: str, interests: list[str]) -> None:
    location = await GeocodingProvider().geocode(city)
    print(f"GEO  {location.display_name[:70]}  ({location.latitude}, {location.longitude})")

    places = await PlacesProvider().find_places(location, interests, limit=12)
    print(f"PLACES {len(places)}\n")
    for place in places:
        print(f"  {place.name[:38]:40} | {place.category:13} | pop {place.popularity:>3} | {place.estimated_hours}h")
        print(f"      {(place.description[:110] or '** NO DESCRIPTION **')}")
        print(f"      img={'yes' if place.image_url else 'no ':3}  coords=({place.latitude:.4f}, {place.longitude:.4f})")


city = sys.argv[1] if len(sys.argv) > 1 else "Rome, Italy"
interests = sys.argv[2].split(",") if len(sys.argv) > 2 else ["history", "culture"]
asyncio.run(main(city, interests))
