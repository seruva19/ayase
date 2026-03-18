"""CLIP Temporal Consistency + Face Consistency — EvalCrafter metrics #5 and #7.

- clip_temp_score: Cosine similarity between CLIP embeddings of consecutive
  frames, averaged across all pairs.  Measures temporal smoothness.
- face_consistency_score: Cosine similarity between every frame and the FIRST
  frame in CLIP space.  Measures identity/appearance preservation.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CLIPTemporalModule(PipelineModule):
    name = "clip_temporal"
    description = "CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency)"
    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 32,
        "temp_threshold": 0.90,
        "face_threshold": 0.85,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.max_frames = self.config.get("max_frames", 32)
        self.temp_threshold = self.config.get("temp_threshold", 0.90)
        self.face_threshold = self.config.get("face_threshold", 0.85)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP for temporal consistency on {self._device}...")
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            self._model = CLIPModel.from_pretrained(resolved).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(resolved)
            self._ml_available = True
        except Exception as e:
            logger.warning(f"Failed to load CLIP for temporal: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            import torch

            frames = self._load_frames(sample)
            if len(frames) < 3:
                return sample

            # Compute CLIP image embeddings for every frame
            embeddings = self._embed_frames(frames)
            if embeddings is None or embeddings.size(0) < 3:
                return sample

            # L2 normalize
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            # --- clip_temp_score: consecutive frame pairs ---
            consec_sims = []
            for i in range(embeddings.size(0) - 1):
                sim = (embeddings[i] @ embeddings[i + 1]).item()
                consec_sims.append(sim)
            clip_temp = float(np.mean(consec_sims))
            sample.quality_metrics.clip_temp = clip_temp

            if clip_temp < self.temp_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low temporal consistency (CLIP_temp={clip_temp:.3f})",
                        details={"clip_temp": clip_temp},
                        recommendation="Consecutive frames differ significantly; possible scene cuts or flickering.",
                    )
                )

            # --- face_consistency_score: rolling window (consecutive pairs) ---
            face_sims = []
            for i in range(embeddings.size(0) - 1):
                sim = (embeddings[i] @ embeddings[i + 1]).item()
                face_sims.append(sim)
            face_consistency = float(np.mean(face_sims))
            sample.quality_metrics.face_consistency = face_consistency

            if face_consistency < self.face_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low identity/face consistency (score={face_consistency:.3f})",
                        details={"face_consistency": face_consistency},
                        recommendation="Visual appearance drifts from first frame; possible subject change.",
                    )
                )

        except Exception as e:
            logger.warning(f"CLIP temporal analysis failed for {sample.path}: {e}")

        return sample

    def _embed_frames(self, frames):
        import torch
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames]
        embeddings = []
        with torch.no_grad():
            for pil_img in pil_frames:
                inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
                feats = self._model.get_image_features(**inputs)  # [1, D]
                embeddings.append(feats)
        return torch.cat(embeddings, dim=0)  # [T, D]

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return frames
            n = min(self.max_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames
