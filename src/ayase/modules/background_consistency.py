"""Background consistency via CLIP pairwise frame similarity (VBench-style).

Computes all-pairs cosine similarity across uniformly sampled frames.
Returns background_consistency (0-1, higher = more consistent). Warns below 0.5."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class BackgroundConsistencyModule(PipelineModule):
    name = "background_consistency"
    description = "Background consistency using CLIP (all pairwise frame similarity)"

    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 16,
        "warning_threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 16)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            logger.info(f"Loading CLIP for Background Consistency on {self._device}...")

            self._model = CLIPModel.from_pretrained(model_name, cache_dir=models_dir, use_safetensors=True).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(model_name, cache_dir=models_dir)
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. Background Consistency disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            import torch
            import torch.nn.functional as F

            embeddings = []

            for frame in frames:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                inputs = self._processor(images=pil_image, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    image_features = extract_features(self._model.get_image_features(**inputs))
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(image_features)

            # All pairwise cosine similarities (VBench style)
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = F.cosine_similarity(embeddings[i], embeddings[j]).item()
                    similarities.append(sim)

            avg_consistency = float(np.mean(similarities))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.background_consistency = avg_consistency

            if avg_consistency < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low background consistency: {avg_consistency:.2f} (Scene might have changed)",
                        details={"consistency_score": avg_consistency},
                    )
                )

        except Exception as e:
            logger.warning(f"Background consistency check failed: {e}")

        return sample

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 2:
                cap.release()
                return []

            n = min(self.max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, n, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames for background consistency: {e}")
        return frames




