"""Checkpoint I/O for the public Pipeline state API."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .models import DatasetStats, Sample

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger("ayase.pipeline")


def save_state(pipeline: "Pipeline", path: Path) -> None:
    """Save current pipeline state to disk for resume."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pipeline_fingerprint": pipeline._pipeline_fingerprint(),
            "results": {k: pipeline._dump_sample_state(k, v) for k, v in pipeline.results.items()},
            "stats": pipeline.stats.model_dump(mode="json"),
            "cache_manifest": {
                k: pipeline._result_manifests.get(k) or pipeline._sample_state_manifest(v)
                for k, v in pipeline.results.items()
            },
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                # default=str is a final backstop for any stray non-JSON
                # value elsewhere so one bad value can't sink the save.
                json.dump(data, f, indent=2, default=str)
            Path(tmp_path).replace(path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        logger.info(f"State saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_state(pipeline: "Pipeline", path: Path) -> None:
    """Load pipeline state from disk."""
    if not path.exists():
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_fingerprint = data.get("pipeline_fingerprint")
        current_fingerprint = pipeline._pipeline_fingerprint()
        saved_stats = None
        if "stats" in data:
            saved_stats = DatasetStats.model_validate(data["stats"])
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return

    if not isinstance(saved_fingerprint, dict):
        pipeline._clear_loaded_state()
        pipeline._start_needs_reset = False
        logger.info("Skipping legacy state file without pipeline fingerprint: %s", path)
        return
    if saved_fingerprint != current_fingerprint:
        pipeline._clear_loaded_state()
        pipeline._start_needs_reset = False
        logger.info("Skipping state file with incompatible pipeline fingerprint: %s", path)
        return

    previous_results = pipeline.results
    previous_signatures = pipeline._result_signatures
    previous_manifests = pipeline._result_manifests
    previous_stats = pipeline.stats.model_copy(deep=True)
    previous_metric_counts = dict(pipeline._metric_counts)
    previous_start_needs_reset = pipeline._start_needs_reset
    previous_restored_checkpoint = pipeline._restored_checkpoint
    pipeline._clear_loaded_state()
    pipeline._start_needs_reset = False
    try:
        results_data = data.get("results", {})
        manifests = data.get("cache_manifest", {})
        partial_restore = False

        if "results" in data:
            for k, v in results_data.items():
                try:
                    context = (
                        {"lip_sync_protocol": pipeline._legacy_lip_sync_protocol}
                        if pipeline._legacy_lip_sync_protocol is not None
                        else None
                    )
                    sample = Sample.model_validate(
                        pipeline._sanitize_cached_sample(k, v), context=context
                    )
                    if (
                        sample.quality_metrics is not None
                        and pipeline._legacy_lip_sync_protocol is not None
                    ):
                        sample.quality_metrics.method_set_lip_sync_legacy_protocol(
                            pipeline._legacy_lip_sync_protocol
                        )
                    manifest = manifests.get(k) if isinstance(manifests, dict) else None
                    if sample.source_video is not None and (
                        not isinstance(manifest, dict) or not manifest.get("source_video")
                    ):
                        partial_restore = True
                        continue
                    if isinstance(manifest, dict):
                        if not pipeline._sample_matches_manifest(manifest):
                            logger.info(
                                f"Skipping stale cache for {k} (state manifest changed)"
                            )
                            partial_restore = True
                            continue
                    elif not pipeline._sample_matches_basic_cache(sample):
                        logger.info(f"Skipping stale cache for {k} (file changed or missing)")
                        partial_restore = True
                        continue
                    pipeline.results[k] = sample
                    pipeline._result_signatures[k] = pipeline._sample_cache_signature(sample)
                    pipeline._result_manifests[k] = (
                        manifest
                        if isinstance(manifest, dict)
                        else pipeline._sample_state_manifest(sample)
                    )
                    pipeline._apply_sample_stats(sample, 1)
                except Exception as e:
                    partial_restore = True
                    logger.warning(f"Failed to load sample {k}: {e}")

        if (
            saved_stats is not None
            and not partial_restore
            and len(pipeline.results) == len(results_data)
        ):
            pipeline._restore_saved_stats(saved_stats)

        pipeline._restored_checkpoint = bool(pipeline.results)
        logger.info(f"State loaded from {path}")
    except Exception as e:
        pipeline.results = previous_results
        pipeline._result_signatures = previous_signatures
        pipeline._result_manifests = previous_manifests
        pipeline.stats = previous_stats
        pipeline._metric_counts = previous_metric_counts
        pipeline._start_needs_reset = previous_start_needs_reset
        pipeline._restored_checkpoint = previous_restored_checkpoint
        logger.error(f"Failed to load state: {e}")
