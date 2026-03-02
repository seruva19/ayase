import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SubjectConsistencyModule(PipelineModule):
    name = "subject_consistency"
    description = "Subject consistency using DINOv2-base (all pairwise frame similarity)"

    default_config = {
        "model_name": "facebook/dinov2-base",
        "max_frames": 16,
        "warning_threshold": 0.6,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 16)
        self.warning_threshold = self.config.get("warning_threshold", 0.6)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = self.config.get("model_name", "facebook/dinov2-base")
            models_dir = self.config.get("models_dir", "models")
            logger.info(f"Loading {model_name} on {self._device}...")

            self._processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=models_dir)
            self._model = AutoModel.from_pretrained(model_name, cache_dir=models_dir, use_safetensors=True).to(self._device)
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. DINO checks disabled.")
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")

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
                    outputs = self._model(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :]
                    emb = F.normalize(emb, p=2, dim=-1)
                    embeddings.append(emb)

            # All pairwise cosine similarities (VBench style)
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = F.cosine_similarity(embeddings[i], embeddings[j]).item()
                    similarities.append(sim)

            avg_consistency = float(np.mean(similarities))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.subject_consistency = avg_consistency

            if avg_consistency < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low subject consistency: {avg_consistency:.2f}",
                        details={"consistency_score": avg_consistency},
                    )
                )

        except Exception as e:
            logger.warning(f"Subject consistency check failed: {e}")

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
            logger.debug(f"Failed to load frames for subject consistency: {e}")
        return frames

