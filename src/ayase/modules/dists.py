"""DISTS (Deep Image Structure and Texture Similarity) module.

DISTS is a full-reference perceptual similarity metric that explicitly
separates structure and texture.  It correlates better with human
perception than SSIM/MS-SSIM for texture-rich content.

Range: 0-1 (lower = more similar / better quality).
Requires a reference video or image.

Uses the ``piq`` package (already an Ayase dependency).
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DISTSModule(PipelineModule):
    name = "dists"
    description = "Deep Image Structure and Texture Similarity (full-reference)"
    default_config = {
        "subsample": 5,  # Process every Nth video frame
        "warning_threshold": 0.3,  # Warn if DISTS > 0.3 (lower is better)
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.3)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._dists_fn = None

    def setup(self) -> None:
        try:
            import torch
            from piq import DISTS as PIQ_DISTS

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            self._dists_fn = PIQ_DISTS().to(self.device)
            self._ml_available = True
            logger.info(f"DISTS module initialised on {self.device}")

        except ImportError:
            logger.warning("piq not installed. Install with: pip install piq")
        except Exception as e:
            logger.warning(f"Failed to setup DISTS: {e}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, img_bgr: np.ndarray):
        """BGR uint8 (H,W,C) → float32 (1,C,H,W) in [0,1]."""
        import torch

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t.to(self.device)

    def _score_pair(self, ref: np.ndarray, dist: np.ndarray) -> Optional[float]:
        """Compute DISTS for one frame-pair."""
        try:
            import torch

            ref_t = self._to_tensor(ref)
            dist_t = self._to_tensor(dist)

            # DISTS requires both inputs to have the same spatial size
            if ref_t.shape != dist_t.shape:
                # Resize distorted to reference size
                h, w = ref_t.shape[2], ref_t.shape[3]
                dist_t = torch.nn.functional.interpolate(
                    dist_t, size=(h, w), mode="bilinear", align_corners=False
                )

            with torch.no_grad():
                score = self._dists_fn(dist_t, ref_t)
            return float(score.item())
        except Exception as e:
            logger.debug(f"DISTS pair computation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # process
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
                score = self._process_video(sample.path, reference)
            else:
                score = self._process_image(sample.path, reference)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dists = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High DISTS (structure/texture dissimilarity): {score:.3f}",
                        details={"dists": score, "threshold": self.warning_threshold},
                        recommendation=(
                            "Content diverges from reference in structure or "
                            "texture. Inspect for hallucinated details or "
                            "texture loss."
                        ),
                    )
                )

            logger.debug(f"DISTS for {sample.path.name}: {score:.4f}")

        except Exception as e:
            logger.error(f"DISTS failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path, ref_path: Path) -> Optional[float]:
        ref = cv2.imread(str(ref_path))
        dist = cv2.imread(str(path))
        if ref is None or dist is None:
            return None
        return self._score_pair(ref, dist)

    def _process_video(self, path: Path, ref_path: Path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0

        while True:
            r1, ref_frame = ref_cap.read()
            r2, dist_frame = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                s = self._score_pair(ref_frame, dist_frame)
                if s is not None:
                    scores.append(s)
            idx += 1

        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None
