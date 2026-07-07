"""T2VScore (Text-to-Video Score) module.

T2VScore (Wu et al., CVPR 2024) evaluates text-to-video generation with two
components computed by trained/prompted models:

  1. Text-Video Alignment (T2VScore-A) — VQA-based: questions are auto-generated
     from the prompt and answered by a video-LLM.
  2. Video Quality (T2VScore-Q) — a mixture-of-experts quality model trained on
     the TVGE human-annotation dataset.

A bare CLIP text-image cosine (alignment) paired with a Laplacian/brightness
sharpness average (quality) is neither of those, so it is not emitted under the
T2VScore name. This module runs only a real T2VScore model configured via
``model_name``; otherwise it reports itself unavailable and leaves
``t2v_score`` / ``t2v_alignment`` / ``t2v_quality`` unset.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class T2VScoreModule(PipelineModule):
    name = "t2v_score"
    description = "Text-to-Video alignment and quality scoring (T2VScore, CVPR 2024)"
    default_config = {
        "model_name": None,  # path/id of a real T2VScore model; CLIP is not T2VScore
        "num_frames": 8,
        "alignment_weight": 0.5,
        "quality_weight": 0.5,
        "device": "auto",
        "warning_threshold": 0.6,
        "trust_remote_code": False,
    }
    metric_groups = {
        "t2v_alignment": "alignment",
        "t2v_quality": "nr_quality",
        "t2v_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", None)
        self.num_frames = self.config.get("num_frames", 8)
        self.alignment_weight = self.config.get("alignment_weight", 0.5)
        self.quality_weight = self.config.get("quality_weight", 0.5)
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = self.config.get("warning_threshold", 0.6)
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        self.device = None
        self._device_str = "cpu"
        self._ml_available = False
        self._backend = None
        self._t2v_model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        model_name = self.model_name
        if not model_name or "clip" in str(model_name).lower():
            self._backend = "unavailable"
            logger.info(
                "T2VScore unavailable: no real T2VScore model configured (a CLIP "
                "backbone is not T2VScore); t2v_score/t2v_alignment/t2v_quality "
                "will not be populated by this module."
            )
            return

        try:
            import torch
            from transformers import AutoModel
            from ayase.runtime import resolve_torch_device

            self._device_str = resolve_torch_device(self.device_config)
            self.device = torch.device(self._device_str)
            self._t2v_model = AutoModel.from_pretrained(
                model_name, trust_remote_code=self.trust_remote_code
            ).to(self.device).eval()
            self._ml_available = True
            self._backend = "t2vscore"
            logger.info("T2VScore loaded real model from %s", model_name)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("T2VScore real model unavailable: %s", e)

    def _load_video_frames(self, video_path, num_frames: int) -> Optional[List[np.ndarray]]:
        try:
            frames = sample_frames(video_path, max_frames=num_frames, color="rgb")
            return frames if frames else None
        except Exception as e:
            logger.debug("Failed to load video frames: %s", e)
            return None

    def _compute_t2v_score(
        self, sample: Sample, caption: str
    ) -> Optional[Tuple[float, float, float]]:
        """Run the real T2VScore model and return (overall, alignment, quality)."""
        import torch

        try:
            frames = self._load_video_frames(sample.path, self.num_frames)
            if frames is None:
                return None

            tensors = []
            for f in frames:
                t = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0
                tensors.append(t)
            clip = torch.stack(tensors).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self._t2v_model(clip, text=[caption])
                if isinstance(output, dict):
                    alignment = float(output["alignment"])
                    quality = float(output["quality"])
                    overall = float(
                        output.get(
                            "score",
                            alignment * self.alignment_weight + quality * self.quality_weight,
                        )
                    )
                elif isinstance(output, (tuple, list)) and len(output) >= 2:
                    overall, alignment = float(output[0]), float(output[1])
                    quality = float(output[2]) if len(output) > 2 else overall
                else:
                    score = float(output.item()) if hasattr(output, "item") else float(output)
                    overall = alignment = quality = score
            return overall, alignment, quality
        except Exception as e:
            logger.warning("T2VScore model inference failed for %s: %s", sample.path, e)
            return None

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "t2vscore":
            return sample
        if not sample.is_video:
            return sample
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            result = self._compute_t2v_score(sample, sample.caption.text)
            if result is None:
                return sample
            t2v_score, alignment, quality = result

            sample.quality_metrics.t2v_score = t2v_score
            sample.quality_metrics.t2v_alignment = alignment
            sample.quality_metrics.t2v_quality = quality

            if t2v_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low T2VScore: {t2v_score:.3f}",
                        details={
                            "t2v_score": t2v_score,
                            "alignment": alignment,
                            "quality": quality,
                            "threshold": self.warning_threshold,
                        },
                        recommendation="Video-text alignment or quality is low. "
                        "Check if video content matches caption and assess technical quality.",
                    )
                )
        except Exception as e:
            logger.warning("T2VScore processing failed for %s: %s", sample.path, e)

        return sample
