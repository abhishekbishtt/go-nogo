#!/usr/bin/env python
"""Demo script showing final route recommendation output."""

from GoNoGo import reason_on_weather_and_recommendation

# Mock routes simulating real API response
test_routes = [
    {
        "route_label": "DEFAULT_ROUTE",
        "duration_minutes": 243.6,
        "distance_km": 344.8,
        "recommendation": "CAUTION",
        "reasoning": "Route contains 2 zones with elevated risk (zones 9, 10); reasons: zone 9: overcast skies, zone 10: limited visibility from dense clouds",
        "weather_zones_reports": []
    },
    {
        "route_label": "DEFAULT_ROUTE_ALTERNATE_1",
        "duration_minutes": 256.7,
        "distance_km": 362.3,
        "recommendation": "CAUTION",
        "reasoning": "Route contains 3 zones with elevated risk (zones 8, 9, 10)",
        "weather_zones_reports": []
    },
    {
        "route_label": "DEFAULT_ROUTE_ALTERNATE_2",
        "duration_minutes": 259.3,
        "distance_km": 369.3,
        "recommendation": "CAUTION",
        "reasoning": "Route contains 3 zones with elevated risk",
        "weather_zones_reports": []
    }
]

recommended = reason_on_weather_and_recommendation(test_routes)

print("\n" + "=" * 70)
print("🎯 FINAL RECOMMENDATION")
print("=" * 70)
print(f"Route Selected: {recommended['route_label']}")
print(f"Status: {recommended['recommendation']}")
print(f"Duration: {recommended['duration_minutes']} minutes")
print(f"Distance: {recommended['distance_km']} km")
print(f"\nReasoning: {recommended['reasoning']}")
print("=" * 70 + "\n")
