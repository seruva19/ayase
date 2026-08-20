"""Lightweight isolated client for Ayase."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ayase_protocol import DatasetStats, QualityMetrics, RemoteModel, Sample, WorkerError

from .pipeline import AyasePipeline

_runtime_init = Path(__file__).resolve().parents[1] / "ayase" / "__init__.py"
if _runtime_init.exists():
    _match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        _runtime_init.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
else:
    _match = None

if _match:
    __version__ = _match.group(1)
else:
    try:
        __version__ = version("ayase")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = [
    "__version__",
    "AyasePipeline",
    "DatasetStats",
    "QualityMetrics",
    "RemoteModel",
    "Sample",
    "WorkerError",
]
