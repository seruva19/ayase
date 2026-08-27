"""NearID reference-image identity similarity using the official ECCV 2026 model.

NearID is trained with near-identity distractors to preserve instance identity
while rejecting visually similar objects in matching contexts. The output is
the raw cosine similarity between the author's L2-normalized embeddings;
higher values indicate stronger identity preservation.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_MODEL_ID = "Aleksandar/nearid-siglip2"
_MODEL_REVISION = "7f69f4a0c753297de708a0217ef32659fe12a008"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class NearIDModule(PipelineModule):
    name = "nearid"
    description = "NearID near-distractor-aware identity similarity (ECCV 2026)"
    default_config = {"model": _MODEL_ID, "device": "auto", "models_dir": "models"}
    models = [
        {
            "id": _MODEL_ID,
            "type": "huggingface",
            "task": "NearID identity-aware SigLIP2 image embeddings",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned to revision {_MODEL_REVISION}",
        }
    ]
    metric_info = {
        "nearid_identity_similarity": (
            "NearID cosine similarity to the reference image (higher=better)"
        ),
    }
    metric_groups = {"nearid_identity_similarity": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self.model_id = str(self.config.get("model", _MODEL_ID))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._torch = None
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            common = {
                "revision": _MODEL_REVISION if self.model_id == _MODEL_ID else None,
                "cache_dir": str(Path(self.models_dir) / "huggingface"),
                "trust_remote_code": True,
            }
            self._processor = AutoImageProcessor.from_pretrained(self.model_id, **common)
            self._model = AutoModel.from_pretrained(self.model_id, **common)
            self._model.eval().to(self._device)
            self._torch = torch
            self._backend = f"nearid:{self.model_id}"
        except Exception as exc:
            self._model = None
            self._processor = None
            logger.warning("NearID setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._model is None or self._processor is None or self._torch is None:
            return sample
        if sample.reference_path is None:
            return sample
        try:
            target = self._load_image(Path(sample.path))
            reference = self._load_image(Path(sample.reference_path))
            if target is None or reference is None:
                return sample
            with self._torch.inference_mode():
                inputs = self._processor(images=[reference, target], return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                embeddings = self._model.get_image_features(**inputs)
                embeddings = self._torch.nn.functional.normalize(embeddings.float(), dim=-1)
                similarity = float((embeddings[0] @ embeddings[1]).detach().cpu())
            if not np.isfinite(similarity):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.nearid_identity_similarity = similarity
        except Exception as exc:
            logger.warning("NearID failed for %s: %s", sample.path, exc)
        return sample

    @staticmethod
    def _load_image(path: Path):
        if path.is_dir():
            path = next(
                (item for item in sorted(path.iterdir()) if item.suffix.lower() in _IMAGE_SUFFIXES),
                None,
            )
            if path is None:
                return None
        with Image.open(path) as image:
            return image.convert("RGB")
