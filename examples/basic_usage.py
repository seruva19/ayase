"""Basic usage example for Ayase."""

from pathlib import Path

from ayase import AyasePipeline

# --- Option 1: High-level API (recommended) ---

ayase = AyasePipeline(modules=["basic"])
results = ayase.run("./my_dataset")

for path, sample in results.items():
    if sample.is_valid:
        print(f"✓ {sample.path.name}")
    else:
        print(f"✗ {sample.path.name}")
        for issue in sample.validation_issues:
            print(f"  - {issue.severity.value}: {issue.message}")

    if sample.quality_metrics:
        print(f"  technical_score={sample.quality_metrics.technical_score}")

print(f"\nTotal: {ayase.stats.total_samples}, Valid: {ayase.stats.valid_samples}")
ayase.export("report.json")


# --- Option 2: Low-level API (full control) ---

import asyncio

from ayase.config import AyaseConfig
from ayase.pipeline import ModuleRegistry, Pipeline
from ayase.scanner import scan_dataset

config = AyaseConfig.load()

ModuleRegistry.discover_modules()
BasicModule = ModuleRegistry.get_module("basic")
pipeline = Pipeline([BasicModule()])
pipeline.start()

samples = scan_dataset(Path("./my_dataset"), recursive=True)
for sample in samples:
    asyncio.run(pipeline.process_sample(sample))

pipeline.stop()
pipeline.export_report(Path("report.json"), format="json")


# --- Option 3: Profile-based ---

from ayase import AyasePipeline

ayase = AyasePipeline(profile={
    "name": "my_quality_check",
    "modules": ["basic", "aesthetic"],
    "module_config": {
        "aesthetic": {"model_name": "openai/clip-vit-large-patch14"},
    },
})
results = ayase.run("./my_dataset")
