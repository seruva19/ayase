"""Standalone JSON entrypoint for isolated AVGen-Bench specialist environments.

Run this file directly: specialist environments do not need Ayase or its host
dependencies. Source-library stdout is diagnostic output, never the IPC channel.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

SOURCE_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
NATIVE_DIMENSIONS = {"Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy"}
REMOTE_DIMENSIONS = {"Text", "Music", "Speech", "Hi-Phy", "Holistic"}
_DLL_HANDLES = []


def _configure_native_libraries() -> None:
    prefix = Path(sys.prefix)
    if os.name != "nt":
        return
    directories = []
    if (prefix / "conda-meta").is_dir():
        directories.append(prefix / "Library" / "bin")
    site = prefix / "Lib" / "site-packages"
    directories.append(site / "torch" / "lib")
    nvidia = site / "nvidia"
    if nvidia.is_dir():
        directories.extend(sorted(nvidia.glob("*/bin")))
        directories.extend(sorted(nvidia.glob("*/lib")))
    for directory in directories:
        if directory.is_dir():
            # Handles must stay alive for the duration of all subsequent imports.
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    for method in ("tolist", "item"):
        converter = getattr(value, method, None)
        if callable(converter):
            return converter()
    raise TypeError(f"non-JSON AVGen-Bench result type: {type(value).__name__}")


def _redact(text: str) -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            text = text.replace(key, "[redacted]")
    return text


def execute(request: Dict[str, Any]) -> Dict[str, Any]:
    dimension = request.get("dimension")
    if (
        type(request.get("schema")) is not int
        or request.get("schema") != 1
        or request.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("AVGen-Bench request schema/source revision mismatch")
    if dimension in NATIVE_DIMENSIONS:
        module_name = "adapters_native"
    elif dimension in REMOTE_DIMENSIONS:
        module_name = "adapters_remote"
    else:
        raise ValueError("unknown AVGen-Bench dimension")
    if dimension == "Face":
        vendor = Path(__file__).resolve().parents[2] / "vendor"
        if not (vendor / "insightface" / "app" / "face_analysis.py").is_file():
            raise RuntimeError("bundled InsightFace runtime is missing")
        sys.path.insert(0, str(vendor))
    # Direct-file loading supplies this directory as sys.path[0], without importing ayase.
    module = importlib.import_module(module_name)
    result = module.run_dimension(request)
    if not isinstance(result, dict):
        raise ValueError("AVGen-Bench adapter returned no result object")
    if result.get("schema") != 1 or result.get("dimension") != dimension:
        raise ValueError("AVGen-Bench adapter returned a mismatched result")
    return result


def main() -> int:
    sys.dont_write_bytecode = True
    _configure_native_libraries()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request: Dict[str, Any] = {}
    try:
        document = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("AVGen-Bench request must be a JSON object")
        request = document
        result = execute(request)
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, default=_json_default)
    except Exception as exc:
        message = _redact(str(exc))[:1200]
        sys.stderr.write(
            _redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        )
        result = {
            "schema": 1,
            "source_revision": SOURCE_REVISION,
            "dimension": request.get("dimension"),
            "error": message,
            "records": [
                {
                    "id": row.get("id"),
                    "status": "failed",
                    "eligible": None,
                    "raw_score": None,
                    "raw_details": {},
                    "error": message,
                }
                for row in request.get("records", [])
                if isinstance(row, dict)
            ],
        }
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, default=_json_default)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_name(args.result.name + f".{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
