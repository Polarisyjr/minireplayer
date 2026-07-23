from __future__ import annotations

import json
import os
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

from minireplay.sdk import llm_identity_headers, remember_llm_attempt

from .patching import patch_method
from .state import state


def _proxy_origins() -> set[tuple[str, str, int | None]]:
    raw = os.environ.get("NATIVE_REPLAY_LLM_PROXY_ORIGINS", "[]")
    values = json.loads(raw)
    if not isinstance(values, list):
        raise RuntimeError("NATIVE_REPLAY_LLM_PROXY_ORIGINS must be a JSON list")
    result = set()
    for value in values:
        parsed = urlsplit(str(value))
        result.add((parsed.scheme, parsed.hostname or "", parsed.port))
    return result


def _matches(url) -> bool:
    parsed = urlsplit(str(url))
    return (parsed.scheme, parsed.hostname or "", parsed.port) in _proxy_origins()


def _upstream_targets() -> list[tuple[str, str]]:
    raw = os.environ.get("NATIVE_REPLAY_UPSTREAM_TARGETS", "{}")
    values = json.loads(raw)
    if not isinstance(values, dict):
        raise RuntimeError("NATIVE_REPLAY_UPSTREAM_TARGETS must be a JSON object")
    result: list[tuple[str, str]] = []
    for target, value in values.items():
        if not isinstance(target, str) or not target or not isinstance(value, str) or not value:
            raise RuntimeError("invalid NATIVE_REPLAY_UPSTREAM_TARGETS entry")
        result.append((value.rstrip("/"), target))
    result.sort(key=lambda item: len(item[0]), reverse=True)
    return result


def _effective_port(parsed) -> int | None:
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _same_endpoint_host(first: str | None, second: str | None) -> bool:
    if first == second:
        return True

    def loopback(value: str | None) -> bool:
        if value == "localhost":
            return True
        try:
            return bool(value) and ip_address(value).is_loopback
        except ValueError:
            return False

    return loopback(first) and loopback(second)


def _redirect(url) -> tuple[str | None, str | None]:
    requested = urlsplit(str(url))
    for base, target in _upstream_targets():
        declared = urlsplit(base)
        declared_path = declared.path.rstrip("/")
        path_matches = requested.path == declared_path or requested.path.startswith(
            f"{declared_path}/"
        )
        if not (
            requested.scheme == declared.scheme
            and _effective_port(requested) == _effective_port(declared)
            and _same_endpoint_host(requested.hostname, declared.hostname)
            and path_matches
        ):
            continue
        proxy = urlsplit(os.environ.get("NATIVE_REPLAY_PROXY_URL", ""))
        if not proxy.scheme or not proxy.netloc:
            raise RuntimeError("replay instrumentation has no LLM proxy URL")
        suffix = requested.path[len(declared_path) :]
        redirected_path = f"{proxy.path.rstrip('/')}{suffix}" or "/"
        return (
            urlunsplit(
                (
                    proxy.scheme,
                    proxy.netloc,
                    redirected_path,
                    requested.query,
                    requested.fragment,
                )
            ),
            target,
        )
    return None, None


def _headers(target: str | None) -> dict[str, str]:
    headers = llm_identity_headers()
    if target is not None:
        headers["X-Native-Replay-Target"] = target
    return headers


def install_http_identity() -> None:
    if not _proxy_origins():
        raise RuntimeError("replay instrumentation requires declared LLM proxy origins")

    try:
        import httpx

        def sync_factory(original):
            def wrapped(self, request, *args, **kwargs):
                redirected, target = _redirect(request.url)
                if redirected is not None:
                    request.url = httpx.URL(redirected)
                intercepted = redirected is not None or _matches(request.url)
                if intercepted:
                    request.headers.update(_headers(target))
                response = original(self, request, *args, **kwargs)
                if intercepted:
                    remember_llm_attempt(response.headers.get("X-Native-Replay-Attempt"))
                return response

            return wrapped

        def async_factory(original):
            async def wrapped(self, request, *args, **kwargs):
                redirected, target = _redirect(request.url)
                if redirected is not None:
                    request.url = httpx.URL(redirected)
                intercepted = redirected is not None or _matches(request.url)
                if intercepted:
                    request.headers.update(_headers(target))
                response = await original(self, request, *args, **kwargs)
                if intercepted:
                    remember_llm_attempt(response.headers.get("X-Native-Replay-Attempt"))
                return response

            return wrapped

        patch_method(httpx.Client, "send", sync_factory)
        patch_method(httpx.AsyncClient, "send", async_factory)
        state().mark("httpx-identity")
        state().mark("httpx-upstream-redirect")
    except ImportError:
        pass

    try:
        import requests

        def requests_factory(original):
            def wrapped(self, request, *args, **kwargs):
                redirected, target = _redirect(request.url)
                if redirected is not None:
                    request.url = redirected
                intercepted = redirected is not None or _matches(request.url)
                if intercepted:
                    request.headers.update(_headers(target))
                response = original(self, request, *args, **kwargs)
                if intercepted:
                    remember_llm_attempt(response.headers.get("X-Native-Replay-Attempt"))
                return response

            return wrapped

        patch_method(requests.Session, "send", requests_factory)
        state().mark("requests-identity")
        state().mark("requests-upstream-redirect")
    except ImportError:
        pass
