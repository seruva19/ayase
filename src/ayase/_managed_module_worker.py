"""Subprocess worker for managed metric modules.

This module is an internal entry point. Protocol data travels through binary pipes. The worker sends backend output
to the log stream before importing the target module.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from copy import deepcopy
from itertools import chain
from typing import Any, BinaryIO, Dict, Iterator, List, Mapping

from ayase._managed_protocol import dump_sample, read_message, write_message
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
    return [dump_sample(sample) for sample in samples]


def _dataset_metrics(pipeline: Pipeline, before: Mapping[str, Any]) -> Dict[str, Any]:
    after = pipeline.stats.model_dump(mode="json")
    return {
        key: value
        for key, value in after.items()
        if before.get(key) != value and key not in {"metric_provenance", "metric_backends"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--qualname", required=True)
    args = parser.parse_args()

    with (
        os.fdopen(os.dup(sys.stdin.fileno()), "rb") as requests,
        os.fdopen(os.dup(sys.stdout.fileno()), "wb") as replies,
        open(os.devnull, "rb") as null_input,
    ):
        os.dup2(null_input.fileno(), sys.stdin.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        if os.name == "nt":
            # Windows subprocesses can inherit Win32 handles independently of CRT fds.
            import ctypes
            import msvcrt
            from ctypes.wintypes import HANDLE

            kernel = ctypes.windll.kernel32
            kernel.SetStdHandle(-10, HANDLE(msvcrt.get_osfhandle(sys.stdin.fileno())))
            kernel.SetStdHandle(-11, HANDLE(msvcrt.get_osfhandle(sys.stdout.fileno())))
        return _serve(args, requests, replies)


def _requests(stream: BinaryIO) -> Iterator[Dict[str, Any]]:
    while True:
        try:
            yield read_message(stream)
        except EOFError:
            return


def _serve(args: argparse.Namespace, requests: BinaryIO, replies: BinaryIO) -> int:
    messages = _requests(requests)
    first = next(messages, None)
    if first is None:
        return 0
    if first.get("operation") != "mount":
        raise ValueError("Managed worker expects mount as its first operation")
    config = dict(first["payload"]["config"])
    config["_ayase_managed_worker"] = True
    module_class = _resolve_class(args.module, args.qualname)
    module = module_class(config)
    pipeline = Pipeline([module])
    module.pipeline = pipeline
    identities: Dict[str, Sample] = {}
    reported_stats = pipeline.stats.model_dump(mode="json")

    for request in chain((first,), messages):
        try:
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
                    "sample": dump_sample(processed),
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
            write_message(replies, {"ok": True, "result": result})
            if operation == "on_dispose":
                return 0
        except Exception as exc:
            write_message(
                replies,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
