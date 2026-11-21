import json

import httpx

BODY = {
    "origin": "Delhi",
    "destination": "Jaipur",
    "start_date": "2026-11-10",
    "end_date": "2026-11-12",
    "travelers": 2,
    "budget_usd": 800,
    "interests": ["history", "architecture"],
    "pace": "packed",
}

response = httpx.post("http://127.0.0.1:8000/api/plan", json=BODY, timeout=180)
print("HTTP", response.status_code)
plan = response.json()

print("destination:", plan["destination_label"])
print("places:", len(plan["places"]))
print("sources:", plan["data_sources"])

for day in plan["itinerary"]:
    print(f"\nDay {day['day']} ({day['date']}) — {day['theme']} — walk {day['walking_distance_km']}km")
    for item in day["items"]:
        print(f"   {item['start_time']}-{item['end_time']}  {item['activity']}")

print("\ntransport:")
for option in plan["travel_options"]:
    print("  ", option["mode"], option["provider"], option["duration_minutes"], "min", option["price_usd"])

print("\nhotels:", len(plan["hotels"]))
for hotel in plan["hotels"][:3]:
    print("  ", hotel["name"], hotel["distance_to_center_km"], "km", hotel["price_per_night_usd"])

print("\ncost:", plan["cost_breakdown"], plan["estimated_cost_usd"])
print("warnings:", json.dumps(plan["warnings"], indent=2))
