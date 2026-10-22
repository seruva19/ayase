"""Focused tests for bounded parallel model URL validation."""

import threading
import urllib.error


class _Response:
    def __init__(self, status=200):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


def _entries(*urls):
    from ayase.models_doc import ModelEntry

    return {
        f"model-{index}": ModelEntry(name=f"model-{index}", source="local", url=url)
        for index, url in enumerate(urls)
    }


def test_validate_urls_success_closes_response(monkeypatch):
    from ayase.models_doc import _validate_urls

    response = _Response()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)

    assert _validate_urls(_entries("https://example.test/model")) == []
    assert response.closed


def test_validate_urls_http_error(monkeypatch):
    from ayase.models_doc import _validate_urls

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert _validate_urls(_entries("https://example.test/missing")) == [
        "  model-0: HTTP 404 — https://example.test/missing"
    ]


def test_validate_urls_network_failure_preserves_sentinel(monkeypatch):
    from ayase.models_doc import _validate_urls

    def fail(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert _validate_urls(_entries("https://example.test/offline")) == ["__NETWORK_UNAVAILABLE__"]


def test_validate_urls_warnings_follow_input_order(monkeypatch):
    from ayase.models_doc import _validate_urls

    gate = threading.Barrier(2)

    def fail(request, timeout):
        gate.wait(timeout=2)
        code = 503 if request.full_url.endswith("first") else 404
        raise urllib.error.HTTPError(request.full_url, code, "failed", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    warnings = _validate_urls(_entries("https://example.test/first", "https://example.test/second"))
    assert warnings == [
        "  model-0: HTTP 503 — https://example.test/first",
        "  model-1: HTTP 404 — https://example.test/second",
    ]
