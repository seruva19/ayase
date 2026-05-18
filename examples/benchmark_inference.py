"""Benchmark Ayase inference settings on a local dataset.

Example:
    python examples/benchmark_inference.py ./videos \
        --modules semantic_alignment,clip_temporal,qclip \
        --batch-sizes 1,2,4,8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from ayase import AyasePipeline
from ayase.config import AyaseConfig
from ayase.scanner import scan_dataset


def _parse_csv_ints(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(max(1, int(part)))
    return sizes or [1]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset directory to scan")
    parser.add_argument(
        "--modules",
        default="semantic_alignment,clip_temporal",
        help="Comma-separated module names",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated sample_batch_size values to compare",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ayase.toml path",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories",
    )
    args = parser.parse_args()

    modules = _parse_csv_strings(args.modules)
    batch_sizes = _parse_csv_ints(args.batch_sizes)
    samples = scan_dataset(args.dataset, recursive=not args.no_recursive)
    if not samples:
        print("No supported media samples found.")
        return 1

    print("batch_size,samples,total_seconds,samples_per_second,module_seconds")
    for batch_size in batch_sizes:
        config = AyaseConfig.load(args.config) if args.config else AyaseConfig.load()
        config.general.sample_batch_size = batch_size
        ayase = AyasePipeline(config=config, modules=modules)

        run_samples = [sample.model_copy(deep=True) for sample in samples]
        started_at = perf_counter()
        ayase.run(args.dataset, samples=run_samples, recursive=not args.no_recursive)
        elapsed = perf_counter() - started_at
        rate = len(run_samples) / elapsed if elapsed else 0.0
        timings = ayase.pipeline.get_timing_report()
        module_seconds = ";".join(
            f"{name}:{data['seconds']:.4f}" for name, data in sorted(timings.items())
        )
        print(
            f"{batch_size},{len(run_samples)},{elapsed:.4f},{rate:.4f},{module_seconds}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
