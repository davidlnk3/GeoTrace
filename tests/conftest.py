import httpx
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries a real HTTP request.

    Clients built on httpx.MockTransport never leave the process, so they are allowed.
    """
    real_send = httpx.AsyncClient.send

    async def refuse(self, request, *args, **kwargs):
        if isinstance(getattr(self, "_transport", None), httpx.MockTransport):
            return await real_send(self, request, *args, **kwargs)
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
