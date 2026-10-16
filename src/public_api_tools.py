"""Read-only public API tools; results remain untrusted and source-linked."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class PublicAPIError(RuntimeError):
    """A public API request failed or returned an unsafe response."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so an allowlisted URL cannot redirect to an untrusted host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class APIObservation:
    provider: str
    data: object
    source_url: str
    retrieved_at: float
    trusted: bool = False


class PublicAPIClient:
    """HTTPS allowlist, bounded response, timeout and basic per-host pacing."""
    ALLOWED_HOSTS = frozenset({
        "api.github.com", "en.wikipedia.org", "geocoding-api.open-meteo.com",
        "api.open-meteo.com", "hn.algolia.com", "api.stackexchange.com",
    })

    def __init__(self, *, timeout=8.0, max_response_bytes=1_000_000, min_interval=0.25, opener=None):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or timeout > 300
            or not math.isfinite(timeout)
        ):
            raise ValueError("timeout must be finite and between 0 and 300 seconds")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 10_000_000:
            raise ValueError("max_response_bytes must be an integer between 1 and 10000000")
        if (
            isinstance(min_interval, bool)
            or not isinstance(min_interval, (int, float))
            or min_interval < 0
            or min_interval > 3600
            or not math.isfinite(min_interval)
        ):
            raise ValueError("min_interval must be finite and between 0 and 3600 seconds")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.min_interval = min_interval
        self._opener = opener or build_opener(_NoRedirectHandler()).open
        self._last_request = {}

    def get_json(self, url, *, provider):
        if not isinstance(url, str) or not url or any(ord(char) < 32 for char in url):
            raise PublicAPIError("URL is malformed")
        try:
            parsed = urlparse(url)
            port = parsed.port
        except (TypeError, ValueError) as exc:
            raise PublicAPIError("URL is malformed") from exc
        if parsed.scheme != "https" or parsed.hostname not in self.ALLOWED_HOSTS or parsed.username or parsed.password or port not in (None, 443):
            raise PublicAPIError("URL is not an allowlisted HTTPS public API endpoint")
        now = time.monotonic()
        previous = self._last_request.get(parsed.hostname)
        if previous is not None:
            delay = self.min_interval - (now - previous)
            if delay > 0:
                time.sleep(delay)
        self._last_request[parsed.hostname] = time.monotonic()
        try:
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "TARA-PublicAPI-Tools/1.0"})
        except (TypeError, ValueError) as exc:
            raise PublicAPIError("URL is malformed") from exc
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise PublicAPIError(f"{provider} response exceeded size limit")
        except HTTPError as exc:
            retry = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code in (403, 429):
                suffix = f"; Retry-After: {retry}" if retry else ""
                raise PublicAPIError(f"{provider} rate limit or access denial{suffix}") from exc
            raise PublicAPIError(f"{provider} returned HTTP {exc.code}") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise PublicAPIError(f"{provider} request failed: {exc}") from exc
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicAPIError(f"{provider} returned invalid JSON") from exc
        if not isinstance(data, (dict, list)):
            raise PublicAPIError(f"{provider} returned an unexpected JSON shape")
        return APIObservation(provider, data, url, time.time(), trusted=False)


class PublicAPITools:
    """Curated read-only queries: GitHub, Wikipedia, Open-Meteo, Hacker News, Stack Exchange."""
    def __init__(self, client=None):
        self.client = client or PublicAPIClient()

    @staticmethod
    def _validate_query_limit(query, limit):
        if not isinstance(query, str):
            raise ValueError("query must be a string containing 1-200 characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer between 1 and 10")
        query = query.strip()
        if not query or len(query) > 200:
            raise ValueError("query must contain 1-200 characters")
        return query

    def search_github_repositories(self, query, *, limit=5):
        query = self._validate_query_limit(query, limit)
        url = "https://api.github.com/search/repositories?" + urlencode({"q": query, "per_page": limit, "sort": "stars"})
        result = self.client.get_json(url, provider="GitHub")
        if not isinstance(result.data, dict) or not isinstance(result.data.get("items"), list):
            raise PublicAPIError("GitHub response is missing repository items")
        return result

    def search_wikipedia(self, query, *, limit=5):
        query = self._validate_query_limit(query, limit)
        url = "https://en.wikipedia.org/w/api.php?" + urlencode({"action": "opensearch", "search": query, "limit": limit, "namespace": 0, "format": "json"})
        result = self.client.get_json(url, provider="Wikipedia")
        if not isinstance(result.data, list) or len(result.data) != 4:
            raise PublicAPIError("Wikipedia response has an unexpected shape")
        return result

    def search_hacker_news(self, query, *, limit=5):
        query = self._validate_query_limit(query, limit)
        url = "https://hn.algolia.com/api/v1/search?" + urlencode({"query": query, "hitsPerPage": limit})
        result = self.client.get_json(url, provider="Hacker News")
        if not isinstance(result.data, dict) or not isinstance(result.data.get("hits"), list):
            raise PublicAPIError("Hacker News response is missing hits")
        return result

    def search_stackoverflow(self, query, *, limit=5):
        query = self._validate_query_limit(query, limit)
        url = "https://api.stackexchange.com/2.3/search/advanced?" + urlencode({
            "order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow", "pagesize": limit,
        })
        result = self.client.get_json(url, provider="Stack Exchange")
        if not isinstance(result.data, dict):
            raise PublicAPIError("Stack Exchange response has an unexpected shape")
        if result.data.get("error_id"):
            raise PublicAPIError("Stack Exchange returned an API error")
        if not isinstance(result.data.get("items"), list):
            raise PublicAPIError("Stack Exchange response is missing items")
        return result

    def get_weather(self, city):
        if not isinstance(city, str):
            raise ValueError("city must be a string containing 1-120 characters")
        city = city.strip()
        if not city or len(city) > 120:
            raise ValueError("city must contain 1-120 characters")
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
        geo = self.client.get_json(geo_url, provider="Open-Meteo geocoding")
        if not isinstance(geo.data, dict) or not isinstance(geo.data.get("results"), list):
            raise PublicAPIError("Geocoding response is missing a results list")
        places = geo.data["results"]
        if not places:
            raise PublicAPIError(f"No geocoding result found for {city!r}")
        place = places[0]
        if not isinstance(place, dict):
            raise PublicAPIError("Geocoding result must be an object")
        try:
            lat, lon = float(place["latitude"]), float(place["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicAPIError("Geocoding response did not contain valid coordinates") from exc
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise PublicAPIError("Geocoding coordinates are out of range")
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode({"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m"})
        weather = self.client.get_json(url, provider="Open-Meteo forecast")
        if not isinstance(weather.data, dict) or not isinstance(weather.data.get("current"), dict):
            raise PublicAPIError("Open-Meteo forecast response is missing current weather data")
        return APIObservation("Open-Meteo", {"location": {k: place.get(k) for k in ("name", "country", "admin1", "latitude", "longitude")}, "weather": weather.data}, weather.source_url, weather.retrieved_at, trusted=False)
