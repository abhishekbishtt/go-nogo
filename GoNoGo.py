from pydantic import BaseModel, ValidationError
from typing import Literal
from groq import Groq
import json
import os
import re
import requests
import dotenv
import polyline
import math

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

MODEL = "llama-3.1-8b-instant"

class final_response(BaseModel):
    location: str
    date: str
    weather_summary: str
    temperature_c: float
    rain_probability: float
    is_daylight: bool
    recommendation: Literal["GO", "NO_GO", "CAUTION"]
    reasoning: str
class weather_response(BaseModel):
    wind_speed: float
    wind_degrees: int
    sunrise: int
    sunset: int
    temp: float
    humidity: int
    min_temp: float
    max_temp: float
    feels_like: float
    cloud_pct: int
class cordinate_response(BaseModel):
    lat: float
    lon: float

class route_cities_response(BaseModel):
    origin: str | None
    destination: str | None
    vehicle: str | None

class vehicle(BaseModel):
    type: Literal["car", "bike", "heavy_vehicle", "pedestrian"]

def weather_tool(city: str, lat_lon: dict) -> dict:
    """Tool to fetch weather data for a given city."""
    api_key = os.getenv("GEO")
    if not api_key:
        return {"error": "API key not found"}
    if not lat_lon:
        lat_lon=get_coordinates(city)
    if not lat_lon:
        return {"error": "City not found"}

    url = f"https://api.api-ninjas.com/v1/weather?lat={lat_lon['lat']}&lon={lat_lon['lon']}"
    headers = {"X-Api-Key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return {"error": "Invalid weather response format", "status": 500}
        
        payload = {
            "wind_speed": data.get("wind_speed"),
            "wind_degrees": data.get("wind_degrees"),
            "sunrise": data.get("sunrise"),
            "sunset": data.get("sunset"),
            "temp": data.get("temp"),
            "humidity": data.get("humidity"),
            "min_temp": data.get("min_temp"),
            "max_temp": data.get("max_temp"),
            "feels_like": data.get("feels_like"),
            "cloud_pct": data.get("cloud_pct"),
            }
        try:
            validated = weather_response(**payload)
            return validated.model_dump()
        except ValidationError:
            return {"error": "Weather response validation failed", "status": 500}
    else:
        return {"error": "Failed to fetch weather data"}

def get_coordinates(city:str)->dict:
    """Get latitude and longitude for a given city using OpenStreetMap Nominatim API."""
    url = f"https://api.api-ninjas.com/v1/geocoding?city={city}"
    headers= {"X-Api-Key": os.getenv("GEO")}    
    raw_cord_response = requests.get(url, headers=headers)
    if raw_cord_response.status_code == 200:
        data= raw_cord_response.json()
        if data:
            try:
                coords= cordinate_response(lat=data[0]["latitude"], lon=data[0]["longitude"])
                return coords.model_dump()
            except ValidationError:
                return None
    return None

def reverse_geocode(lat: float, lon: float) -> str | None:
    """Gets the city or place name for a given latitude and longitude using API Ninjas Reverse Geocoding API."""
    api_key = os.getenv("GEO")
    if not api_key:
        return None
    url = f"https://api.api-ninjas.com/v1/reversegeocoding?lat={lat}&lon={lon}"
    headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                return data[0].get("name")
    except Exception:
        pass
    return None

def _detect_vehicle_from_text(prompt: str) -> str | None:
    """Detect a vehicle from explicit Indian-English transport keywords."""
    text = prompt.lower()

    if re.search(r"\b(?:bicycle|cycling|cycle|pedal cycle|cycle ride)\b", text):
        return "BICYCLE"

    if re.search(r"\b(?:bike|biking|motorbike|motorcycle|scooty|two wheeler|scooty ride)\b", text):
        return "TWO_WHEELER"

    if re.search(r"\b(?:car|drive|driving|truck|van|auto)\b", text):
        return "DRIVE"

    if re.search(r"\b(?:walk|walking|pedestrian|foot)\b", text):
        return "WALK"

    return None


def extract_route_cities_and_vehicle(prompt: str) -> tuple[str | None, str | None, str | None]:
    """
    Extracts route information using the GROQ API (Llama 3.1) or falls back to a local parser.
    Includes Indian English localization for vehicle types via system prompt instructions.
    """
    # Try using GROQ model if API key is available; otherwise use local fallback parser.
    detected_vehicle = _detect_vehicle_from_text(prompt)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_extract(prompt)

    try:
        groq_client = Groq(api_key=api_key)
        MODEL = "llama-3.1-8b-instant"

        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the origin city, destination city, and mode of transport from the user's travel query. "
                        "Respond ONLY with a valid JSON object containing exactly three keys: 'origin', 'destination', and 'vehicle'. "
                        "The 'vehicle' value must strictly match one of these Google Routes API modes: "
                        "'DRIVE' (car/truck/heavy vehicle), 'WALK' (pedestrian), 'BICYCLE' (non-motorized), or 'TWO_WHEELER' (motorized). "
                        "Do not output any other vehicle values. "
                        "In Indian English, the word 'bike' always means a motorized two-wheeler, not a bicycle. "
                        "If the prompt contains any of the exact terms: bike, biking, motorbike, motorcycle, scooty, scooty ride, or two wheeler, the vehicle MUST be TWO_WHEELER. "
                        "If the prompt contains any of the exact terms: bicycle, cycling, cycle, pedal cycle, or cycle ride, the vehicle MUST be BICYCLE. "
                        "If both bike and cycle/bicycle appear, choose BICYCLE only when the prompt explicitly says 'cycle' or 'bicycle'. Otherwise choose TWO_WHEELER. "
                        "NEVER map the exact token 'bike' to BICYCLE. "
                        "Examples: 'bike from Delhi to Agra' -> TWO_WHEELER; 'bicycle from Delhi to Agra' -> BICYCLE; 'cycle from Pune to Mumbai' -> BICYCLE; 'motorbike from Delhi to Agra' -> TWO_WHEELER. "
                        "If the mode of transport is not specified, return null for the 'vehicle' key. "
                        "If a city cannot be identified, return null for that key."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}  # CRITICAL: Forces Llama to output raw JSON without markdown
        )

        response_message = response.choices[0].message
        content = getattr(response_message, "content", "") or ""

        if not content.strip():
            return _fallback_extract(prompt)

        try:
            cities_data = json.loads(content)
            validated_cities = route_cities_response(**cities_data)
            vehicle = detected_vehicle or validated_cities.vehicle
            return validated_cities.origin, validated_cities.destination, vehicle

        except (json.JSONDecodeError, ValidationError):
            return _fallback_extract(prompt)

    except Exception:
        return _fallback_extract(prompt)

def _fallback_extract(prompt: str) -> tuple[str | None, str | None, str | None]:
    """
    Simple heuristic parser for origin, destination, and vehicle used when the model is unavailable.
    
    This function uses regular expressions to extract routing information. 
    It includes localized mapping for Indian English (e.g., 'bike' maps to a 
    motorized TWO_WHEELER, and 'cycle' maps to BICYCLE).

    Args:
        prompt (str): The user's input text requesting a route.

    Returns:
        tuple[str | None, str | None, str | None]: A tuple containing:
            - origin (str | None): The starting location, or None if not found.
            - destination (str | None): The ending location, or None if not found.
            - vehicle (str | None): The mode of transport ('DRIVE', 'BICYCLE', 
              'TWO_WHEELER', 'WALK'), or None if not specified.
    """
    text = prompt.lower()

    # Vehicle detection tailored for Indian English usage
    vehicle = None
    if re.search(r"\b(car|drive|driving|truck|van|auto)\b", text):
        vehicle = "DRIVE"
    elif re.search(r"\b(motorcycle|motorbike|scooter|bike|scooty)\b", text):
        # In India, "bike" and "scooty" universally refer to motorized two-wheelers
        vehicle = "TWO_WHEELER"
    elif re.search(r"\b(bicycle|cycling|cycle)\b", text):
        # "Cycle" is the standard Indian term for a non-motorized bicycle
        vehicle = "BICYCLE"
    elif re.search(r"\b(walk|walking|pedestrian|foot)\b", text):
        vehicle = "WALK"

    # Basic origin/destination patterns: "from X to Y" or "X to Y"
    origin = None
    destination = None
    
    # Match "from [Origin] to [Destination]" and stop before trailing transport qualifiers.
    m = re.search(
        r"\bfrom\s+(.+?)\s+(?:to|->|-)\s+(.+?)(?:\s+(?:on|by|with|via|using)\b|$)",
        text,
    )
    if not m:
        # Match "[Origin] to [Destination]" with the same trailing stop words.
        m = re.search(
            r"\b(.+?)\s+to\s+(.+?)(?:\s+(?:on|by|with|via|using)\b|$)",
            text,
        )

    if m:
        # Clean up extra spaces and convert to Title Case
        origin = " ".join(m.group(1).split()).title()
        destination = " ".join(m.group(2).split()).title()
    else:
        # Look for capitalized words as a fallback destination
        cities = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", prompt)
        if len(cities) == 1:
            destination = cities[0]

    return origin, destination, vehicle

def get_routes_with_alternatives(lat_lon_origin: tuple[float, float], lat_lon_destination: tuple[float, float], mode_of_transport: str) -> list[dict] | None:
    """
    Fetches driving routes from Google Routes API including alternative paths.
    Returns the raw JSON response containing up to 4 routes.
    """
    # Best practice: Keep your API key in environment variables
    api_key = os.getenv("GMAP_API_KEY") 
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # CRITICAL: routeLabels tells you which is the default vs alternative
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.routeLabels"
    }
    
    if not api_key:
        return None

    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": lat_lon_origin[0], 
                    "longitude": lat_lon_origin[1]
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": lat_lon_destination[0], 
                    "longitude": lat_lon_destination[1]
                }
            }
        },
        "travelMode": mode_of_transport,
        "routingPreference": "TRAFFIC_AWARE",
        
        # This is the exact toggle that requests alternative paths
        "computeAlternativeRoutes": True
    }
    
    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status() 
        raw_data = response.json()
        extracted_routes = []
        for route in raw_data.get("routes", []):
            encode = route.get("polyline", {}).get("encodedPolyline", "")
            coords = polyline.decode(encode) if encode else []
            extracted_routes.append({
                "routeLabels": route.get("routeLabels", ["UNKNOWN"])[0],
                "duration_seconds": float(route.get('duration', '0').replace('s', '')),
                "distance_km": route.get('distanceMeters', 0) / 1000,
                "coordinates": coords
            })
        return extracted_routes

    except requests.exceptions.RequestException:
        return None

def _extract_precipitation(weather: dict) -> float:
    """Return the best available precipitation-like value from a weather payload."""
    if not isinstance(weather, dict):
        return 0.0

    for key in ("rain_probability", "precip", "precipitation", "precip_mm", "precip_prob", "rain", "rainfall"):
        value = weather.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _assess_zone_severity(weather: dict) -> tuple[str, str]:
    """Categorize a single weather zone as GO, CAUTION, or NO_GO and return a reason."""
    if not isinstance(weather, dict):
        return "CAUTION", "weather data unavailable"

    wind_speed = float(weather.get("wind_speed") or 0)
    humidity = float(weather.get("humidity") or 0)
    cloud_pct = float(weather.get("cloud_pct") or 0)
    rain_probability = _extract_precipitation(weather)

    severe_reasons = []
    caution_reasons = []

    if wind_speed >= 50:
        severe_reasons.append("strong wind")
    elif wind_speed >= 35:
        caution_reasons.append("high wind")

    if rain_probability >= 60:
        severe_reasons.append("heavy rain potential")
    elif rain_probability >= 30:
        caution_reasons.append("rain likely")

    if humidity >= 95:
        severe_reasons.append("very high humidity")
    elif humidity >= 85:
        caution_reasons.append("elevated humidity")

    if cloud_pct >= 90:
        caution_reasons.append("limited visibility from dense clouds")
    elif cloud_pct >= 75:
        caution_reasons.append("overcast skies")

    if severe_reasons:
        return "NO_GO", ", ".join(severe_reasons)
    if caution_reasons:
        return "CAUTION", ", ".join(caution_reasons)
    return "GO", "weather conditions look favorable"


def _assess_route_safety(route_report: dict, origin_city: str = "", destination_city: str = "") -> tuple[str, str]:
    """Analyze all weather zones for a route and return a recommendation plus reasoning in natural language."""
    zones = route_report.get("weather_zones_reports", [])
    if not zones:
        return "CAUTION", "Weather data unavailable"

    # Evaluate recommendation category and reasons for each zone
    zone_details = []
    has_nogo = False
    has_caution = False

    for zone in zones:
        weather = zone.get("weather")
        category, reason = _assess_zone_severity(weather)
        if category == "NO_GO":
            has_nogo = True
        elif category == "CAUTION":
            has_caution = True
        
        zone_details.append({
            "idx": zone["zone_index"],
            "lat": zone["latitude"],
            "lon": zone["longitude"],
            "category": category,
            "reason": reason
        })

    # Overall recommendation
    if has_nogo:
        recommendation = "NO_GO"
    elif has_caution:
        recommendation = "CAUTION"
    else:
        recommendation = "GO"

    # Group consecutive zones with the same category and reason
    segments = []
    if zone_details:
        current_segment = [zone_details[0]]
        for zd in zone_details[1:]:
            if zd["category"] == current_segment[-1]["category"] and zd["reason"] == current_segment[-1]["reason"]:
                current_segment.append(zd)
            else:
                segments.append(current_segment)
                current_segment = [zd]
        segments.append(current_segment)

    # Describe warning segments (skip GO segments)
    segment_descriptions = []
    for seg in segments:
        first = seg[0]
        last = seg[-1]
        category = first["category"]
        reason = first["reason"]

        if category == "GO":
            continue

        if origin_city and first["idx"] == 0:
            start_place = origin_city
        else:
            start_place = reverse_geocode(first["lat"], first["lon"]) or f"Zone {first['idx']}"

        if destination_city and last["idx"] == len(zones) - 1:
            end_place = destination_city
        else:
            end_place = reverse_geocode(last["lat"], last["lon"]) or f"Zone {last['idx']}"

        if start_place == end_place:
            segment_descriptions.append(f"around {start_place} ({reason})")
        else:
            segment_descriptions.append(f"between {start_place} and {end_place} ({reason})")

    if not segment_descriptions:
        reasoning = "No significant weather hazards were detected along this route."
    else:
        reasoning = f"Route contains hazards: {', '.join(segment_descriptions)}."

    return recommendation, reasoning


def process_routes_into_weather_zones(decoded_routes: list[dict], origin_city: str = "", destination_city: str = "", interval_km: float=10) -> list[dict]:
    """
    Given an array of decoded routes, fetch weather data for each path divided into zones using the haversine formula.
    """
    final_app_payload = []
    for route in decoded_routes:
        dense_coords = route.get("coordinates", [])
        if not dense_coords:
            continue

        sampled_zones = [dense_coords[0]]  # Always include the start point
        accumulated_distance = 0.0
        for i in range(1, len(dense_coords)):
            prev = dense_coords[i - 1]
            curr = dense_coords[i]
            segment_distance = haversine_distance(prev, curr)
            accumulated_distance += segment_distance
            if accumulated_distance >= interval_km:
                sampled_zones.append(curr)
                accumulated_distance = 0.0

        if dense_coords[-1] not in sampled_zones:
            sampled_zones.append(dense_coords[-1])

        weather_zones_reports = []
        for idx, (lat, lon) in enumerate(sampled_zones):
            weather_zones_reports.append(
                {
                    "zone_index": idx,
                    "latitude": lat,
                    "longitude": lon,
                    "weather": weather_tool(lat_lon={"lat": lat, "lon": lon}, city="")
                }
            )

        recommendation, reasoning = _assess_route_safety({
            "weather_zones_reports": weather_zones_reports
        }, origin_city=origin_city, destination_city=destination_city)

        final_app_payload.append({
            "route_label": route.get("routeLabels", "UNKNOWN"),
            "duration_minutes": round(route.get("duration_seconds", 0) / 60, 1),
            "distance_km": round(route.get("distance_km", 0), 2),
            "recommendation": recommendation,
            "reasoning": reasoning,
            "coordinates": route.get("coordinates", []),
            "weather_zones_reports": weather_zones_reports
        })

    return final_app_payload


def reason_on_weather_and_recommendation(routes: list[dict]) -> dict:
    """Select and return the best route based on weather conditions."""
    if not isinstance(routes, list) or not routes:
        return {
            "recommendation": "CAUTION",
            "reasoning": "No routes available."
        }
    
    # Prioritize routes: GO > CAUTION > NO_GO, and prefer shorter duration within same priority
    go_routes = [r for r in routes if r.get("recommendation") == "GO"]
    caution_routes = [r for r in routes if r.get("recommendation") == "CAUTION"]
    nogo_routes = [r for r in routes if r.get("recommendation") == "NO_GO"]
    
    # Select best route from highest priority group
    best_routes = go_routes or caution_routes or nogo_routes
    best_route = min(best_routes, key=lambda r: r.get("duration_minutes", float('inf')))
    
    return {
        "route_label": best_route.get("route_label"),
        "recommendation": best_route.get("recommendation", "CAUTION"),
        "reasoning": best_route.get("reasoning", "Weather analysis unavailable"),
        "duration_minutes": best_route.get("duration_minutes"),
        "distance_km": best_route.get("distance_km"),
        "coordinates": best_route.get("coordinates", []),
        "weather_zones_reports": best_route.get("weather_zones_reports", [])
    }

def via_point(all_extracted_routes: list, best_route: dict) -> str:
    """
    Finds a coordinate on the best route near the midpoint that is not near other routes.
    Returns it as a 'latitude,longitude' string.
    """
    best_coords = best_route.get("coordinates", [])
    if not best_coords:
        return ""

    # Get label of the best route to exclude it
    best_label = best_route.get("route_label") or best_route.get("routeLabels")

    # Collect coordinates from all other routes
    other_coords = []
    for route in all_extracted_routes:
        route_label = route.get("route_label") or route.get("routeLabels")
        if route_label != best_label:
            other_coords.extend(route.get("coordinates", []))

    # If there are no other routes, the midpoint is automatically the best choice
    mid_idx = len(best_coords) // 2
    if not other_coords:
        lat, lon = best_coords[mid_idx]
        return f"{lat},{lon}"

    # Search outward from the midpoint index: [mid, mid-1, mid+1, mid-2, mid+2, ...]
    indices = sorted(range(len(best_coords)), key=lambda i: abs(i - mid_idx))

    # Try different distance thresholds from 5.0 km down to 0.5 km
    for threshold in [5.0, 3.0, 2.0, 1.0, 0.5]:
        for idx in indices:
            candidate = best_coords[idx]
            
            # Check if this candidate is far enough from all coordinates in other routes
            is_distinct = True
            for other_c in other_coords:
                if haversine_distance(candidate, other_c) < threshold:
                    is_distinct = False
                    break
            
            if is_distinct:
                return f"{candidate[0]},{candidate[1]}"

    # Fallback to the physical midpoint if no distinct point could be found
    lat, lon = best_coords[mid_idx]
    return f"{lat},{lon}"




def haversine_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def run_conversation(user_prompt):
    """Run a conversation with travel and weather route recommendation logic."""
    origin_city, destination_city, mode_of_transport = extract_route_cities_and_vehicle(user_prompt)
    if not mode_of_transport:
        mode_of_transport = "DRIVE"

    if not origin_city and not destination_city:
        return "Could not extract origin or destination cities from the query. Please rephrase your question with clear city names."

    lat_lon_origin = get_coordinates(origin_city) if origin_city else None
    lat_lon_destination = get_coordinates(destination_city) if destination_city else None

    if not lat_lon_origin or not lat_lon_destination:
        return "Could not find coordinates for both cities. Please check the city names and try again."

    routes_data = get_routes_with_alternatives(
        (lat_lon_origin["lat"], lat_lon_origin["lon"]),
        (lat_lon_destination["lat"], lat_lon_destination["lon"]),
        mode_of_transport
    )
    if not routes_data:
        return "Failed to fetch route data. Please try again later."

    route_weather = process_routes_into_weather_zones(routes_data, origin_city=origin_city, destination_city=destination_city)
    if not route_weather:
        return "Could not compute weather zones for the route data."

    best_route = reason_on_weather_and_recommendation(route_weather)
    vp_coord = via_point(route_weather, best_route)
    
    via_city = None
    if vp_coord:
        try:
            vplat, vplon = map(float, vp_coord.split(','))
            via_city = reverse_geocode(vplat, vplon)
        except ValueError:
            pass

    via_str = f" via {via_city}" if via_city else ""
    summary_text = (
        f"Route: {origin_city} to {destination_city}{via_str} ({best_route.get('route_label', 'UNKNOWN')})\n"
        f"Recommendation: {best_route.get('recommendation')}\n"
        f"Reasoning: {best_route.get('reasoning')}"
    )

    return {
        "origin": origin_city,
        "destination": destination_city,
        "transport_mode": mode_of_transport,
        "via_point_coordinate": vp_coord,
        "via_point_name": via_city,
        "recommendation": best_route.get("recommendation"),
        "reasoning": best_route.get("reasoning"),
        "summary": summary_text,
        "routes": route_weather
    }
    



if __name__ == "__main__":
    prompt = "I want to go from dehradun to guptkashi by motor bike."
    result = run_conversation(prompt)
    if isinstance(result, dict):
        print("\n=== FINAL OUTPUT ===")
        print(result["summary"])
        print(f"Via Point Coordinate: {result['via_point_coordinate']}")
        print(f"Via Point Name: {result['via_point_name']}")
    else:
        print(result)