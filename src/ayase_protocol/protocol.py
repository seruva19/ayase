"""Versioned Ayase wire contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1
HEALTH_PATH = "/v1/health"
RUN_PATH = "/v1/run"
EXPORT_PATH = "/v1/export"
SHUTDOWN_PATH = "/v1/shutdown"


class WorkerError(RuntimeError):
    """The worker rejected or failed a request."""


def validate_health(health: Dict[str, Any], expected_ayase: Optional[str] = None) -> None:
    if health.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("incompatible Ayase worker protocol")
    actual = health.get("ayase_version")
    if expected_ayase is not None and actual != expected_ayase:
        raise RuntimeError(
            f"Ayase worker version {actual} does not match client version {expected_ayase}"
        )
