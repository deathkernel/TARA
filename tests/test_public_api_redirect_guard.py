"""Regression checks for the public API client's network boundary."""

from src.public_api_tools import PublicAPIClient, _NoRedirectHandler


def test_default_client_installs_redirect_rejection_handler():
    client = PublicAPIClient()
    opener = client._opener.__self__

    assert any(isinstance(handler, _NoRedirectHandler) for handler in opener.handlers)
