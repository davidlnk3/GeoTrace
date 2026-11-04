"""Turn coordinates into a place name, using OpenStreetMap's Nominatim service.

Nominatim's usage policy asks for an identifying User-Agent and no more than one request per
second, so this module sends a descriptive User-Agent and spaces out its requests. It sends
only the coordinates; it does not need an account or an API key.
"""
import threading
import time

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "GeoTrace/0.1 (https://github.com/davidlnk3/GeoTrace)"
MIN_INTERVAL_S = 1.0          # Nominatim allows at most one request per second

_lock = threading.Lock()
_last_request = 0.0


def coordinates_to_location(latitude: float, longitude: float, zoom: int = 10) -> dict:
    """Return the place name and address fields for a latitude and longitude in degrees.

    `zoom` sets the level of detail: 3 is a country, 5 a region, 10 a city, 18 a building.
    Fields that Nominatim does not know are None.
    """
    global _last_request

    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180].")

    params = {
        "format": "jsonv2",
        "lat": latitude,
        "lon": longitude,
        "zoom": zoom,
        "addressdetails": 1,
    }
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        response = httpx.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT},
                             timeout=15)
        _last_request = time.monotonic()
    response.raise_for_status()

    data = response.json()
    if "error" in data:
        raise ValueError(f"No location found: {data['error']}")

    address = data.get("address", {})
    return {
        "latitude": latitude,
        "longitude": longitude,
        "display_name": data.get("display_name"),
        "country": address.get("country"),
        "country_code": address.get("country_code"),
        "region": address.get("state") or address.get("region") or address.get("province"),
        "city": (address.get("city") or address.get("town") or address.get("village")
                 or address.get("municipality")),
    }
