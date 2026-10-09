"""Estimate audio/video offset for videos that have audio metadata.

The only backend is Synchformer (Iashin et al., ICASSP 2024) via the vendored
inferencer. It writes ``av_sync_offset`` in milliseconds using the model's
native convention — positive means audio leads video. If Synchformer cannot
be loaded the module emits no score; an energy-envelope cross-correlation
heuristic is not a substitute for the learned model.

An absolute offset above ``warning_threshold_ms`` adds a warning.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioVisualSyncModule(PipelineModule):
    name = "av_sync"
    provenance = "published"
    sources = {
        "av_sync_offset": "Synchformer (Iashin et al., ICASSP 2024) — https://github.com/v-iashin/Synchformer",
    }
    description = "Audio-video synchronisation offset detection (Synchformer)"
    default_config = {
        "warning_threshold_ms": 80.0,  # Warn if |offset| > 80 ms
    }
    models = [
        {
            "id": "Synchformer",
            "type": "local",
            "task": "Optional learned A/V offset backend when local weights are configured",
        },
    ]
    metric_info = {
        "av_sync_offset": "Estimated audio-video offset in milliseconds",
    }
    metric_groups = {
        "av_sync_offset": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold_ms", 80.0)
        self._ml_available = False
        self._backend = None
        self._syncformer = None

    _WEIGHTS_URLS = {
        "24-01-04T16-39-21.pt": (
            "https://huggingface.co/AkaneTendo25/ayase-assets/resolve/main/"
            "synchformer/24-01-04T16-39-21.pt"
        ),
    }

    def _ensure_synchformer_weights(self, model_path: Path) -> bool:
        """Pre-position ``<model_path>/24-01-04T16-39-21.pt`` from the Ayase
        HF mirror. The vendored ``SyncformerInferencer`` does NOT auto-download
        — it expects the file to be in place. The original CSC Finland URL
        (a3s.fi) is unreliable on some networks; the HF mirror is preferred.
        Returns True iff weights are present."""
        import os
        target = Path(model_path) / "24-01-04T16-39-21.pt"
        if target.exists() and target.stat().st_size > 1_000_000:
            return True
        url = self._WEIGHTS_URLS.get("24-01-04T16-39-21.pt")
        if not url:
            return False
        try:
            from ayase.config import download_model_file, resolve_assets_url
            download_model_file(
                "24-01-04T16-39-21.pt",
                resolve_assets_url(url, self.config),
                str(model_path),
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Synchformer weights download failed: %s", e)
            return False
        return target.exists() and target.stat().st_size > 1_000_000

    def setup(self) -> None:
        try:
            vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verse_bench"
            sys.path.insert(0, str(vendor_root))
            from syncformer.syncformer_inferencer import SyncformerInferencer

            models_dir = Path(self.config.get("models_dir", "models"))
            model_path = Path(self.config.get("syncformer_model_path", models_dir / "syncformer"))
            model_path.mkdir(parents=True, exist_ok=True)
            if not self._ensure_synchformer_weights(model_path):
                self._backend = "unavailable"
                logger.warning(
                    "Synchformer weights unavailable (cannot download from "
                    "AkaneTendo25/ayase-assets HF mirror)"
                )
                return
            self._syncformer = SyncformerInferencer(str(model_path))
            self._ml_available = True
            self._backend = "syncformer"
            logger.info("AudioVisualSync initialised with Synchformer backend")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("Synchformer backend unavailable: %s", e)

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        # Check that the video has an audio stream
        if sample.audio_metadata is None:
            return sample

        offset = self._compute_syncformer(sample.path)
        if offset is not None:
            return self._store_offset(sample, offset)
        return sample

    def _compute_syncformer(self, video_path: Path) -> Optional[float]:
        try:
            offset_sec = float(self._syncformer.infer(str(video_path)))
            return offset_sec * 1000.0
        except Exception as e:
            logger.warning("Synchformer inference failed for %s: %s", video_path, e)
            return None

    def _store_offset(self, sample: Sample, offset_ms: float) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.av_sync_offset = offset_ms

        if abs(offset_ms) > self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"A/V sync offset: {offset_ms:+.1f} ms",
                    details={
                        "offset_ms": offset_ms,
                        "threshold_ms": self.warning_threshold,
                    },
                    recommendation=(
                        "Audio and video are noticeably out of sync. "
                        "Check muxing, frame rate conversion, or "
                        "audio processing pipeline."
                    ),
                )
            )
        return sample


class AudioVisualSyncCompatModule(AudioVisualSyncModule):
    """Compatibility alias matching filename-based discovery."""

    name = "audio_visual_sync"
