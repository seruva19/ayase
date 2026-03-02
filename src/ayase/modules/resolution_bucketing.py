"""Resolution / aspect-ratio bucketing validator.

Validates that video/image resolution and aspect ratio fit into standard
training buckets used by video diffusion models (Wan, HunyuanVideo,
CogVideoX, etc.).  Flags samples that would require extreme padding,
cropping, or distortion to fit any standard bucket.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Common training resolution buckets (width, height) used by video diffusion models.
# Sources: Wan 2.1, HunyuanVideo, CogVideoX, SVD documentation.
DEFAULT_BUCKETS: List[Tuple[int, int]] = [
    # Landscape
    (1280, 720), (1024, 576), (960, 544), (832, 480), (768, 432),
    (640, 360), (512, 288), (512, 320),
    # Portrait
    (720, 1280), (576, 1024), (544, 960), (480, 832), (432, 768),
    (360, 640), (288, 512), (320, 512),
    # Square
    (1024, 1024), (768, 768), (512, 512), (256, 256),
]


class ResolutionBucketingModule(PipelineModule):
    name = "resolution_bucketing"
    description = "Validates resolution/aspect-ratio fit for training buckets"
    default_config = {
        "max_crop_ratio": 0.15,   # Max fraction of pixels lost to crop
        "max_scale_factor": 2.0,  # Max upscale allowed before flagging
        "divisibility": 8,        # Resolution must be divisible by this
        "min_resolution": 256,    # Minimum dimension (either axis)
        "buckets": None,          # Custom bucket list, or use defaults
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_crop_ratio = self.config.get("max_crop_ratio", 0.15)
        self.max_scale_factor = self.config.get("max_scale_factor", 2.0)
        self.divisibility = self.config.get("divisibility", 8)
        self.min_resolution = self.config.get("min_resolution", 256)
        custom = self.config.get("buckets")
        self.buckets = [tuple(b) for b in custom] if custom else DEFAULT_BUCKETS

    def process(self, sample: Sample) -> Sample:
        w = sample.width
        h = sample.height
        if not w or not h:
            return sample

        # --- Divisibility ---
        if w % self.divisibility != 0 or h % self.divisibility != 0:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Resolution {w}x{h} not divisible by {self.divisibility}",
                    details={"width": w, "height": h, "divisibility": self.divisibility},
                    recommendation=f"Crop or pad to nearest multiple of {self.divisibility}.",
                )
            )

        # --- Minimum resolution ---
        if min(w, h) < self.min_resolution:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Resolution too low ({w}x{h}), min dimension < {self.min_resolution}",
                    details={"width": w, "height": h},
                    recommendation="Discard or upscale (but upscaled content trains poorly).",
                )
            )

        # --- Best bucket fit ---
        best_bucket, crop_ratio, scale_factor = self._find_best_bucket(w, h)
        if best_bucket is None:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"No suitable training bucket for {w}x{h}",
                    details={"width": w, "height": h},
                )
            )
            return sample

        bw, bh = best_bucket

        if crop_ratio > self.max_crop_ratio:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Best bucket {bw}x{bh} requires {crop_ratio:.1%} crop from {w}x{h}",
                    details={
                        "source": f"{w}x{h}",
                        "bucket": f"{bw}x{bh}",
                        "crop_ratio": float(crop_ratio),
                    },
                    recommendation="Consider center-cropping or resizing. Excessive crop may lose content.",
                )
            )

        if scale_factor > self.max_scale_factor:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Would need {scale_factor:.1f}x upscale to fit bucket {bw}x{bh}",
                    details={
                        "source": f"{w}x{h}",
                        "bucket": f"{bw}x{bh}",
                        "scale_factor": float(scale_factor),
                    },
                    recommendation="Source is too small. Upscaling introduces blur; prefer discarding.",
                )
            )

        return sample

    def _find_best_bucket(
        self, w: int, h: int
    ) -> Tuple[Optional[Tuple[int, int]], float, float]:
        """Find the bucket requiring least crop/scale.  Returns (bucket, crop_ratio, scale_factor)."""
        src_ar = w / h
        best = None
        best_score = float("inf")
        best_crop = 0.0
        best_scale = 1.0

        for bw, bh in self.buckets:
            bucket_ar = bw / bh

            # Scale source so that it covers the bucket (fit-cover)
            if src_ar > bucket_ar:
                # Source is wider: match height, crop width
                scale = bh / h
                scaled_w = w * scale
                crop_pixels = max(0, scaled_w - bw) * bh
                total_pixels = bw * bh
            else:
                # Source is taller: match width, crop height
                scale = bw / w
                scaled_h = h * scale
                crop_pixels = max(0, scaled_h - bh) * bw
                total_pixels = bw * bh

            crop_ratio = crop_pixels / total_pixels if total_pixels > 0 else 0
            scale_factor = 1.0 / scale if scale > 0 else float("inf")  # >1 means upscale

            # Combined score: penalize both crop and extreme scaling
            ar_diff = abs(math.log(src_ar / bucket_ar)) if bucket_ar > 0 else 99
            score = crop_ratio + ar_diff * 0.5

            if score < best_score:
                best_score = score
                best = (bw, bh)
                best_crop = crop_ratio
                best_scale = scale_factor

        return best, best_crop, best_scale
