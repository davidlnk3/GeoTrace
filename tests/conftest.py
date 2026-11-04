import httpx
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries a real HTTP request."""

    async def refuse(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
