import json
from urllib.error import HTTPError

import pytest

from src.public_api_tools import APIObservation, PublicAPIClient, PublicAPIError, PublicAPITools, _NoRedirectHandler


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit=-1):
        return self.body[:limit]


def fake_client(payload, **kwargs):
    body = json.dumps(payload).encode()
    return PublicAPIClient(opener=lambda request, timeout: FakeResponse(body), min_interval=0, **kwargs)


def test_client_returns_untrusted_source_linked_observation():
    client = fake_client({"ok": True})
    result = client.get_json("https://api.github.com/example", provider="GitHub")
    assert isinstance(result, APIObservation)
    assert result.data == {"ok": True}
    assert result.source_url == "https://api.github.com/example"
    assert result.trusted is False


@pytest.mark.parametrize("url", [
    "http://api.github.com/repos",
    "https://example.com/data",
    "https://user:pass@api.github.com/repos",
    "https://api.github.com:444/repos",
])
def test_client_rejects_non_allowlisted_or_unsafe_urls(url):
    client = fake_client({})
    with pytest.raises(PublicAPIError):
        client.get_json(url, provider="test")


def test_client_rejects_oversized_response():
    client = fake_client({"long": "x" * 50}, max_response_bytes=10)
    with pytest.raises(PublicAPIError, match="size limit"):
        client.get_json("https://api.github.com/example", provider="GitHub")


def test_client_rejects_invalid_json():
    client = PublicAPIClient(opener=lambda request, timeout: FakeResponse(b"not json"), min_interval=0)
    with pytest.raises(PublicAPIError, match="invalid JSON"):
        client.get_json("https://api.github.com/example", provider="GitHub")


def test_github_search_validates_response_and_limits():
    tools = PublicAPITools(fake_client({"items": []}))
    assert tools.search_github_repositories("python").data["items"] == []
    with pytest.raises(ValueError):
        tools.search_github_repositories("python", limit=11)
    with pytest.raises(ValueError):
        tools.search_github_repositories("  ")


def test_wikipedia_rejects_unexpected_shape():
    tools = PublicAPITools(fake_client({"not": "a list"}))
    with pytest.raises(PublicAPIError, match="unexpected shape"):
        tools.search_wikipedia("TARA")


def test_weather_validates_coordinates_and_returns_untrusted_data():
    responses = [
        {"results": [{"name": "Pune", "country": "India", "latitude": 18.52, "longitude": 73.85}]},
        {"current": {"temperature_2m": 30}},
    ]
    def opener(request, timeout):
        return FakeResponse(json.dumps(responses.pop(0)).encode())
    tools = PublicAPITools(PublicAPIClient(opener=opener, min_interval=0))
    result = tools.get_weather("Pune")
    assert result.data["location"]["name"] == "Pune"
    assert result.data["weather"]["current"]["temperature_2m"] == 30
    assert result.trusted is False


def test_hacker_news_search_validates_hits_and_uses_allowlisted_host():
    tools = PublicAPITools(fake_client({"hits": [{"title": "TARA"}]}))
    result = tools.search_hacker_news("TARA")
    assert result.data["hits"][0]["title"] == "TARA"
    assert result.provider == "Hacker News"
    assert result.trusted is False
    assert "hn.algolia.com" in result.source_url
    with pytest.raises(PublicAPIError, match="missing hits"):
        PublicAPITools(fake_client({"items": []})).search_hacker_news("TARA")


def test_stackoverflow_search_validates_items_and_query_limits():
    tools = PublicAPITools(fake_client({"items": [{"title": "Python question"}]}))
    result = tools.search_stackoverflow("python")
    assert result.data["items"][0]["title"] == "Python question"
    assert result.provider == "Stack Exchange"
    assert result.trusted is False
    assert "api.stackexchange.com" in result.source_url
    with pytest.raises(ValueError):
        tools.search_stackoverflow("x" * 201)
    with pytest.raises(PublicAPIError, match="missing items"):
        PublicAPITools(fake_client({"error_id": 400})).search_stackoverflow("python")


def test_redirect_handler_rejects_redirects():
    handler = _NoRedirectHandler()
    request = __import__("urllib.request", fromlist=["Request"]).Request("https://api.github.com/example")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://example.com/steal") is None


def test_stackoverflow_surfaces_provider_error_before_missing_items():
    tools = PublicAPITools(fake_client({"error_id": 400, "error_name": "bad_parameter"}))
    with pytest.raises(PublicAPIError, match="API error"):
        tools.search_stackoverflow("python")
