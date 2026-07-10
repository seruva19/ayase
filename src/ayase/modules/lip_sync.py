"""Lip Sync Error — LSE-D / LSE-C (SyncNet, reference-free).

Measures audio-visual lip synchronisation of a single clip:
  LSE-D (Lip Sync Error - Distance): lower = better sync
  LSE-C (Lip Sync Error - Confidence): higher = better sync

These are SyncNet-defined metrics computed directly on the clip's own
audio-visual streams — no reference video and no benchmark dataset are
required. The backend is the self-contained SyncNet implementation bundled
with the toolkit; the ``syncnet_v2.model`` weights are fetched from the model
mirror on first use (cached under ``models_dir/lip_sync/``). If the weights
cannot be fetched, the clip has no audio, or no talking face is detected, the
scores are left ``None`` (an audio/mouth-energy heuristic is NOT a stand-in).
"""

import logging
import sys
from pathlib import Path

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Weights are mirrored under <models_dir>/lip_sync/ and fetched on first use,
# matching the other weight-backed modules.
_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
_SYNCNET_REL = "lip_sync/syncnet_v2.model"


class LipSyncModule(PipelineModule):
    name = "lip_sync"
    description = "LSE-D/LSE-C lip sync error (SyncNet, reference-free; no dataset required)"
    default_config = {
        "models_dir": "models",   # weights land under models_dir/lip_sync/
        "device": "auto",
    }
    metric_info = {
        "lse_c": "SyncNet lip-sync confidence (higher=better)",
        "lse_d": "SyncNet lip-sync distance (lower=better)",
    }
    metric_groups = {
        "lse_c": "temporal",
        "lse_d": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._inferencer = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # The self-contained SyncNet implementation is bundled under the
        # verse_bench vendor tree; add its root to the import path so the
        # ``syncnet`` package resolves to it (not an external install).
        vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verse_bench"
        if not vendor_root.exists():
            logger.warning("Lip Sync: bundled SyncNet not found at %s; LSE left unset", vendor_root)
            return
        vendor_root_str = str(vendor_root)
        if vendor_root_str not in sys.path:
            sys.path.insert(0, vendor_root_str)

        try:
            from syncnet.syncnet_inferencer import SyncnetInferencer
        except Exception as e:
            logger.warning("Lip Sync: SyncNet backend import failed (%s); LSE left unset", e)
            return

        # Fetch the SyncNet weights from the mirror (cached afterwards), same as
        # the other weight-backed modules.
        from ayase.config import download_model_file

        models_dir = str(self.config.get("models_dir", "models"))
        try:
            weight_path = download_model_file(_SYNCNET_REL, _MODELS_BASE + _SYNCNET_REL, models_dir)
        except Exception as e:
            logger.warning("Lip Sync: could not fetch syncnet_v2.model (%s); LSE left unset", e)
            return

        # SyncnetInferencer loads ``<model_dir>/syncnet_v2.model``, so hand it
        # the directory the weight was cached into.
        model_dir = str(Path(weight_path).parent)
        try:
            self._inferencer = SyncnetInferencer(model_dir)
        except Exception as e:
            logger.warning("Lip Sync: SyncNet inferencer init failed (%s); LSE left unset", e)
            return

        self._ml_available = True
        self._backend = "syncnet"
        logger.info("Lip Sync module initialised (bundled SyncNet, mirrored weights)")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if not self._ml_available or self._inferencer is None:
            return sample

        try:
            # infer -> (offset, conf, dists); conf is LSE-C, min(dists) is LSE-D.
            offset, conf, dists = self._inferencer.infer(str(sample.path))
        except Exception as e:
            logger.error("Lip Sync failed on %s: %s", sample.path.name, e)
            return sample

        if conf is None or dists is None or len(dists) == 0:
            logger.debug("Lip Sync: no talking face / audio-visual sync in %s; LSE left unset", sample.path)
            return sample

        lse_c = float(conf)
        lse_d = float(min(dists))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.lse_c = lse_c
        sample.quality_metrics.lse_d = lse_d
        logger.debug("Lip Sync %s: LSE-C=%.3f LSE-D=%.3f", sample.path.name, lse_c, lse_d)
        return sample
