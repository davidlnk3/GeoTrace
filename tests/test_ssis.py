from geotrace.core import ssis as ssis_module
from geotrace.core.ssis import SSISClient


def _record_client(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, space_id, token=None, **kwargs):
            calls.append({"space_id": space_id, "token": token})

    monkeypatch.setattr(ssis_module, "Client", FakeClient)
    return calls


def test_anonymous_when_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = _record_client(monkeypatch)
    SSISClient()._connect()
    assert calls == [{"space_id": ssis_module.DEFAULT_SPACE, "token": None}]


def test_hf_token_env_is_sent(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    calls = _record_client(monkeypatch)
    SSISClient()._connect()
    assert calls[0]["token"] == "hf_test"


def test_explicit_token_wins_over_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    calls = _record_client(monkeypatch)
    SSISClient(token="hf_explicit")._connect()
    assert calls[0]["token"] == "hf_explicit"


def test_client_is_built_once(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = _record_client(monkeypatch)
    client = SSISClient()
    client._connect()
    client._connect()
    assert len(calls) == 1
