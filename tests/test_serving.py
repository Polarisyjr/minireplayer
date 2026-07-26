from __future__ import annotations

import urllib.error

import pytest

from minireplay import serving
from minireplay.errors import InfrastructureError


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def test_wait_serving_ready_checks_every_model_endpoint(monkeypatch) -> None:
    observed: list[str] = []

    def urlopen(url: str, *, timeout: int):
        observed.append(url)
        assert timeout == 3
        return _Response()

    monkeypatch.setattr(serving.urllib.request, "urlopen", urlopen)

    serving.wait_serving_ready(
        ["http://127.0.0.1:8000", "http://127.0.0.1:8001/"],
        timeout_s=0,
    )

    assert set(observed) == {
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8001/v1/models",
    }


def test_wait_serving_ready_rejects_a_listening_but_unready_endpoint(monkeypatch) -> None:
    def urlopen(_url: str, *, timeout: int):
        assert timeout == 3
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(serving.urllib.request, "urlopen", urlopen)

    with pytest.raises(InfrastructureError, match="did not become API-ready"):
        serving.wait_serving_ready(["http://127.0.0.1:8000"], timeout_s=0)
