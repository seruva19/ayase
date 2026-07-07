"""VADER --- Video Diffusion Alignment via Reward Gradients (ICLR 2025).

GitHub: https://github.com/mihirp1998/VADER

VADER is a method for fine-tuning video diffusion models via reward
gradients.  It uses reward models internally --- most notably HPS v2
(Human Preference Score).

This module extracts the HPS v2 reward signal that VADER relies on. HPS v2
is the real backend; if the ``hpsv2`` package is not installed the module
reports itself unavailable rather than substituting a CLIP-aesthetic proxy
(which is not the HPS-v2 reward VADER uses).

vader_score --- higher = better (0-1 range, normalised HPS v2 reward).
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VADERModule(PipelineModule):
    name = "vader"
    description = "VADER HPS v2 reward signal (ICLR 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-large-patch14",
    }
    metric_groups = {
        "vader_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Real backend: HPS v2 via the ``hpsv2`` package.
        if self._try_hpsv2_setup():
            return

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "VADER unavailable: the hpsv2 package (HPS v2 reward) is not installed; "
            "vader_score will not be populated by this module."
        )

    def _try_hpsv2_setup(self) -> bool:
        """Try HPS v2 via the ``hpsv2`` package."""
        try:
            import hpsv2  # noqa: F401
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = hpsv2
            self._ml_available = True
            self._backend = "hpsv2"
            logger.info("VADER (HPS v2 package) initialised")
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("HPS v2 package setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            score = self._process_hpsv2(sample)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vader_score = score
            logger.debug("VADER for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("VADER failed for %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------
    # HPS v2 scoring
    # ------------------------------------------------------------------

    def _process_hpsv2(self, sample: Sample) -> Optional[float]:
        """Score using HPS v2 package.  Returns normalised 0-1 score."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = arrays_to_pil(frames)

        # hpsv2.score expects (images, prompt) -> list of floats
        prompt = "a high quality image"
        scores = []
        for pil_img in pil_frames:
            try:
                result = self._model.score(pil_img, prompt)
                if isinstance(result, (list, tuple)):
                    raw = float(result[0])
                else:
                    raw = float(result)
                # HPS v2 raw scores are typically in ~0.18-0.34 range;
                # linearly rescale the real reward to 0-1.
                normalised = float(np.clip((raw - 0.18) / 0.16, 0.0, 1.0))
                scores.append(normalised)
            except Exception as e:
                logger.debug("HPS v2 frame scoring failed: %s", e)

        if not scores:
            return None
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VADER frame loading failed for %s: %s", sample.path, e)
            return []
