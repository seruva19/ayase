"""Subprocess worker for managed metric modules.

This module is an internal entry point. Protocol data travels through JSON files;
stdout and stderr are reserved for the worker log.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ayase.models import QualityMetrics, Sample, observe_metric_writes
from ayase.pipeline import Pipeline


def _resolve_class(module_name: str, qualname: str) -> type:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not isinstance(value, type):
        raise TypeError(f"Managed module target is not a class: {module_name}.{qualname}")
    return value


def _copy_sample(target: Sample, document: Mapping[str, Any]) -> Sample:
    updated = Sample.model_validate(document)
    for field in Sample.model_fields:
        setattr(target, field, deepcopy(getattr(updated, field)))
    return target


def _samples(records: List[Dict[str, Any]], identities: Dict[str, Sample]) -> List[Sample]:
    samples: List[Sample] = []
    for record in records:
        token = str(record["token"])
        sample = identities.get(token)
        if sample is None:
            sample = Sample.model_validate(record["sample"])
            identities[token] = sample
        else:
            _copy_sample(sample, record["sample"])
        samples.append(sample)
    return samples


def _dump_samples(samples: List[Sample]) -> List[Dict[str, Any]]:
    return [sample.model_dump(mode="json") for sample in samples]


def _dataset_metrics(pipeline: Pipeline, before: Mapping[str, Any]) -> Dict[str, Any]:
    after = pipeline.stats.model_dump(mode="json")
    return {
        key: value
        for key, value in after.items()
        if before.get(key) != value and key not in {"metric_provenance", "metric_backends"}
    }


def _reply(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _read_request(path: Path) -> Dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while True:
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--qualname", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["_ayase_managed_worker"] = True
    module_class = _resolve_class(args.module, args.qualname)
    module = module_class(config)
    pipeline = Pipeline([module])
    module.pipeline = pipeline
    identities: Dict[str, Sample] = {}
    reported_stats = pipeline.stats.model_dump(mode="json")

    for line in sys.stdin:
        request_name = line.strip()
        if not request_name:
            continue
        request_path = Path(request_name)
        request: Dict[str, Any] = {}
        try:
            request = _read_request(request_path)
            reply_path = Path(request["reply"])
            operation = request["operation"]
            payload = request.get("payload", {})
            if operation == "mount":
                module.on_mount()
                backend = getattr(module, "_backend", "not-declared")
                ml_available = getattr(module, "_ml_available", "not-declared")
                unavailable = backend in {None, "unavailable"} or ml_available is False
                result: Dict[str, Any] = {
                    "mounted": bool(module._mounted) and not unavailable,
                    "backend": backend,
                    "ml_available": ml_available,
                }
            elif operation == "on_execute":
                module.on_execute()
                result = {}
            elif operation == "process":
                record = payload["record"]
                sample = _samples([record], identities)[0]
                written: Dict[int, set[str]] = {}
                with observe_metric_writes(
                    lambda metrics, field: written.setdefault(id(metrics), set()).add(field)
                ):
                    processed = module.process(sample)
                identities[str(record["token"])] = processed
                result = {
                    "sample": processed.model_dump(mode="json"),
                    "written_fields": sorted(
                        written.get(id(processed.quality_metrics), set())
                        if isinstance(processed.quality_metrics, QualityMetrics)
                        else ()
                    ),
                }
            elif operation == "process_batch":
                records = payload["records"]
                written = {}
                with observe_metric_writes(
                    lambda metrics, field: written.setdefault(id(metrics), set()).add(field)
                ):
                    processed_samples = module.process_batch(_samples(records, identities))
                for record, sample in zip(records, processed_samples):
                    identities[str(record["token"])] = sample
                result = {
                    "samples": _dump_samples(processed_samples),
                    "written_fields": [
                        (
                            sorted(written.get(id(sample.quality_metrics), set()))
                            if isinstance(sample.quality_metrics, QualityMetrics)
                            else []
                        )
                        for sample in processed_samples
                    ],
                }
            elif operation == "post_process":
                samples = _samples(payload["records"], identities)
                written = {}
                with observe_metric_writes(
                    lambda metrics, field: written.setdefault(id(metrics), set()).add(field)
                ):
                    module.post_process(samples)
                result = {
                    "samples": _dump_samples(samples),
                    "written_fields": [
                        (
                            sorted(written.get(id(sample.quality_metrics), set()))
                            if isinstance(sample.quality_metrics, QualityMetrics)
                            else []
                        )
                        for sample in samples
                    ],
                }
            elif operation == "on_dispose":
                module.on_dispose()
                result = {}
            else:
                raise ValueError(f"Unknown managed worker operation: {operation}")
            result["dataset_metrics"] = _dataset_metrics(pipeline, reported_stats)
            reported_stats = pipeline.stats.model_dump(mode="json")
            _reply(reply_path, {"ok": True, "result": result})
            if operation == "on_dispose":
                return 0
        except Exception as exc:
            reply_path = Path(request.get("reply", request_path.with_suffix(".reply.json")))
            _reply(
                reply_path,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
