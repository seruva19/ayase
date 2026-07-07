"""MS-SWD — Multiscale Sliced Wasserstein Distance (ECCV 2024).

Full-reference perceptual colour-difference metric. Ayase runs the real
pretrained ``MS_SWD_learned`` model via ``pyiqa`` (a genuine full-reference
metric that compares two images and returns a perceptual colour distance;
lower = more similar).

As a dataset-level batch metric this module aggregates the real per-pair MS-SWD
distance:
  * when paired references are provided, distorted-vs-reference distances are
    averaged;
  * otherwise the mean MS-SWD between consecutive samples is reported as an
    inter-sample colour-divergence signal.

Backend: **pyiqa** ``msswd`` metric (real pretrained weights).

msswd_score — LOWER = better (closer / more consistent colour distributions)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.image import load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule
from ayase.runtime import resolve_torch_device

logger = logging.getLogger(__name__)


class MSSWDModule(BatchMetricModule):
    name = "msswd"
    description = "MS-SWD multiscale sliced Wasserstein colour distance via pyiqa (batch, lower=better)"
    default_config = {
        "device": "auto",
        "max_side": 512,  # cap frame long side to bound MS-SWD cost
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_side = self.config.get("max_side", 512)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import pyiqa

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("msswd", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("MS-SWD (pyiqa msswd) initialised on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("MS-SWD unavailable: pyiqa not installed (pip install pyiqa)")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("MS-SWD unavailable: %s", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Return the representative RGB frame as a contiguous uint8 array."""
        if not self._ml_available:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        # load_representative_frame may hand back a read-only cache view; copy so
        # downstream resize/tensor ops are safe.
        return self._cap_side(np.ascontiguousarray(frame))

    def _cap_side(self, rgb: np.ndarray) -> np.ndarray:
        max_side = int(self.max_side)
        if max_side <= 0:
            return rgb
        h, w = rgb.shape[:2]
        longer = max(h, w)
        if longer <= max_side:
            return rgb
        scale = max_side / longer
        return cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, rgb: np.ndarray):
        import torch

        return (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        ).to(self._device)

    def _pair_distance(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> Optional[float]:
        """Real MS-SWD distance between two RGB frames (resized to match)."""
        import torch

        h, w = a_rgb.shape[:2]
        if b_rgb.shape[:2] != (h, w):
            b_rgb = cv2.resize(b_rgb, (w, h), interpolation=cv2.INTER_AREA)
        try:
            with torch.no_grad():
                return float(self._model(self._to_tensor(a_rgb), self._to_tensor(b_rgb)).item())
        except Exception as e:
            logger.debug("MS-SWD pair scoring failed: %s", e)
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Aggregate real MS-SWD distances across the accumulated frames."""
        if not self._ml_available or not features:
            return None

        distances: List[float] = []
        if reference_features:
            n = min(len(features), len(reference_features))
            for i in range(n):
                d = self._pair_distance(features[i], reference_features[i])
                if d is not None:
                    distances.append(d)
        else:
            for i in range(len(features) - 1):
                d = self._pair_distance(features[i], features[i + 1])
                if d is not None:
                    distances.append(d)

        if not distances:
            return None
        return float(np.mean(distances))
