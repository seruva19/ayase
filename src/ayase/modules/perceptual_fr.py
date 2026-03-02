"""Advanced Full-Reference Perceptual Metrics module.

Computes several well-established full-reference image quality metrics
in a single pass:

  FSIM  — Feature Similarity Index (phase congruency + gradient)
          Range 0-1, higher = better.
  GMSD  — Gradient Magnitude Similarity Deviation
          Range 0+, lower = better (0 = identical).
  VSI   — Visual Saliency-weighted Index
          Range 0-1, higher = better.

All three are available in the ``piq`` package (already an Ayase
dependency).  They are full-reference metrics — they require a
reference image / video.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PerceptualFRModule(PipelineModule):
    name = "perceptual_fr"
    description = "FSIM + GMSD + VSI full-reference perceptual metrics"
    default_config = {
        "subsample": 5,  # Every Nth video frame
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._fsim_fn = None
        self._gmsd_fn = None
        self._vsi_fn = None

    def setup(self) -> None:
        try:
            import torch
            import piq

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            self._fsim_fn = piq.fsim
            self._gmsd_fn = piq.gmsd
            self._vsi_fn = piq.vsi
            self._ml_available = True
            logger.info(f"Perceptual FR metrics (FSIM/GMSD/VSI) initialised on {self.device}")

        except ImportError:
            logger.warning("piq not installed. Install with: pip install piq")
        except Exception as e:
            logger.warning(f"Failed to setup perceptual FR metrics: {e}")

    # ------------------------------------------------------------------
    def _to_tensor(self, img_bgr: np.ndarray):
        import torch

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

    def _score_pair(
        self, ref: np.ndarray, dist: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (fsim, gmsd, vsi) for one frame pair."""
        try:
            import torch

            ref_t = self._to_tensor(ref)
            dist_t = self._to_tensor(dist)

            # Align sizes
            if ref_t.shape != dist_t.shape:
                h, w = ref_t.shape[2], ref_t.shape[3]
                dist_t = torch.nn.functional.interpolate(
                    dist_t, size=(h, w), mode="bilinear", align_corners=False
                )

            with torch.no_grad():
                fsim_val = self._fsim_fn(dist_t, ref_t, data_range=1.0).item()
                gmsd_val = self._gmsd_fn(dist_t, ref_t, data_range=1.0).item()
                vsi_val = self._vsi_fn(dist_t, ref_t, data_range=1.0).item()

            return fsim_val, gmsd_val, vsi_val

        except Exception as e:
            logger.debug(f"Perceptual FR pair failed: {e}")
            return None, None, None

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            if sample.is_video:
                fsim, gmsd, vsi = self._process_video(sample.path, reference)
            else:
                ref_img = cv2.imread(str(reference))
                dist_img = cv2.imread(str(sample.path))
                if ref_img is None or dist_img is None:
                    return sample
                fsim, gmsd, vsi = self._score_pair(ref_img, dist_img)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            if fsim is not None:
                sample.quality_metrics.fsim = fsim
            if gmsd is not None:
                sample.quality_metrics.gmsd = gmsd
            if vsi is not None:
                sample.quality_metrics.vsi_score = vsi

            logger.debug(
                f"Perceptual FR for {sample.path.name}: "
                f"FSIM={fsim:.3f} GMSD={gmsd:.4f} VSI={vsi:.3f}"
                if fsim is not None
                else f"Perceptual FR: no scores"
            )

        except Exception as e:
            logger.error(f"Perceptual FR failed for {sample.path}: {e}")

        return sample

    def _process_video(
        self, path: Path, ref_path: Path
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))

        fsim_vals, gmsd_vals, vsi_vals = [], [], []
        idx = 0

        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                f, g, v = self._score_pair(ref_f, dist_f)
                if f is not None:
                    fsim_vals.append(f)
                if g is not None:
                    gmsd_vals.append(g)
                if v is not None:
                    vsi_vals.append(v)
            idx += 1

        ref_cap.release()
        dist_cap.release()

        return (
            float(np.mean(fsim_vals)) if fsim_vals else None,
            float(np.mean(gmsd_vals)) if gmsd_vals else None,
            float(np.mean(vsi_vals)) if vsi_vals else None,
        )
