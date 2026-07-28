"""Learned image-to-image similarity with DINOv2, CLIP, SigLIP, and LPIPS.

All backends use their upstream pretrained weights and download them
automatically into ``models_dir``. No proxy or heuristic score is emitted when
a backend is unavailable. Redundant embedding transforms and alternate LPIPS
trunks are deliberately omitted.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

LEARNED_FIELDS = (
    "i2i_dinov2_cls_similarity",
    "i2i_dinov2_patch_similarity",
    "i2i_clip_similarity",
    "i2i_siglip_similarity",
    "i2i_lpips_alex",
)


class I2ILearnedModule(PipelineModule):
    """Compute five complementary learned similarities for an image pair."""

    name = "i2i_learned"
    description = "DINOv2, CLIP, SigLIP, and LPIPS image-to-image fidelity"
    default_config = {
        "dinov2_model": "facebook/dinov2-small",
        "clip_model": "openai/clip-vit-base-patch32",
        "siglip_model": "google/siglip-base-patch16-224",
        "models_dir": "models",
        "device": "auto",
    }
    required_packages = ["torch", "transformers", "lpips"]
    models = [
        {
            "id": "facebook/dinov2-small",
            "type": "huggingface",
            "task": "Global and patch-level I2I representation fidelity",
            "auto_download": True,
        },
        {
            "id": "openai/clip-vit-base-patch32",
            "type": "huggingface",
            "task": "CLIP image embedding similarity",
            "auto_download": True,
        },
        {
            "id": "google/siglip-base-patch16-224",
            "type": "huggingface",
            "task": "SigLIP image embedding similarity",
            "auto_download": True,
        },
        {
            "id": "lpips",
            "type": "pip_package",
            "install": "pip install lpips",
            "task": "AlexNet LPIPS weights",
            "auto_download": True,
        },
    ]
    metric_info = {
        "i2i_dinov2_cls_similarity": "Cosine similarity of DINOv2 CLS embeddings (-1 to 1)",
        "i2i_dinov2_patch_similarity": "Mean aligned-patch DINOv2 cosine similarity (-1 to 1)",
        "i2i_clip_similarity": "Cosine similarity of CLIP image embeddings (-1 to 1)",
        "i2i_siglip_similarity": "Cosine similarity of SigLIP image embeddings (-1 to 1)",
        "i2i_lpips_alex": "LPIPS distance with AlexNet trunk (lower=better)",
    }
    metric_groups = {field: "fr_quality" for field in LEARNED_FIELDS}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._torch: Any = None
        self._device: Any = None
        self._dinov2: Any = None
        self._dinov2_processor: Any = None
        self._clip: Any = None
        self._clip_processor: Any = None
        self._siglip: Any = None
        self._siglip_processor: Any = None
        self._lpips_alex: Any = None
        self._backend: Optional[str] = None

    def setup(self) -> None:
        try:
            import torch

            self._torch = torch
            configured = self.config.get("device", "auto")
            self._device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if configured == "auto"
                else torch.device(configured)
            )
        except Exception as exc:
            logger.warning("i2i_learned requires torch: %s", exc)
            return

        cache_dir = str(Path(self.config.get("models_dir", "models")).resolve())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # LPIPS constructs torchvision trunks internally. Point torch.hub at the
        # same configured cache so every automatically downloaded weight remains
        # under models_dir, just like the Hugging Face backends above.
        self._torch.hub.set_dir(str(Path(cache_dir) / "torch"))
        loaded = []
        try:
            from transformers import AutoImageProcessor, AutoModel

            model_id = self.config.get("dinov2_model", "facebook/dinov2-small")
            self._dinov2_processor = AutoImageProcessor.from_pretrained(
                model_id, cache_dir=cache_dir
            )
            self._dinov2 = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
            self._dinov2.eval().to(self._device)
            loaded.append("dinov2")
        except Exception as exc:
            logger.warning("DINOv2 I2I backend unavailable: %s", exc)
        try:
            from transformers import CLIPModel, CLIPProcessor

            model_id = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self._clip = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
            self._clip.eval().to(self._device)
            loaded.append("clip")
        except Exception as exc:
            logger.warning("CLIP I2I backend unavailable: %s", exc)
        try:
            from transformers import AutoProcessor, SiglipModel

            model_id = self.config.get("siglip_model", "google/siglip-base-patch16-224")
            self._siglip_processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self._siglip = SiglipModel.from_pretrained(model_id, cache_dir=cache_dir)
            self._siglip.eval().to(self._device)
            loaded.append("siglip")
        except Exception as exc:
            logger.warning("SigLIP I2I backend unavailable: %s", exc)
        try:
            import lpips

            self._lpips_alex = lpips.LPIPS(net="alex", version="0.1").eval().to(self._device)
            loaded.append("lpips_alex")
        except Exception as exc:
            logger.warning("LPIPS I2I backends unavailable: %s", exc)
        self._backend = "+".join(loaded) if loaded else None

    def _cosine(self, left: Any, right: Any) -> float:
        return float(self._torch.nn.functional.cosine_similarity(left, right, dim=-1).mean().item())

    def process(self, sample: Sample) -> Sample:
        reference = getattr(sample, "reference_path", None)
        if reference is None or sample.is_video or self._torch is None:
            return sample
        reference = Path(reference)
        if not reference.is_file() or not sample.path.is_file():
            return sample
        try:
            from PIL import Image

            generated = Image.open(sample.path).convert("RGB")
            target = Image.open(reference).convert("RGB")
            values: Dict[str, float] = {}
            with self._torch.inference_mode():
                if self._dinov2 is not None:
                    inputs = self._dinov2_processor(
                        images=[generated, target], return_tensors="pt"
                    ).to(self._device)
                    hidden = self._dinov2(**inputs).last_hidden_state
                    values["i2i_dinov2_cls_similarity"] = self._cosine(hidden[0, 0], hidden[1, 0])
                    left = self._torch.nn.functional.normalize(hidden[0, 1:], dim=-1)
                    right = self._torch.nn.functional.normalize(hidden[1, 1:], dim=-1)
                    values["i2i_dinov2_patch_similarity"] = float((left * right).sum(-1).mean().item())
                if self._clip is not None:
                    inputs = self._clip_processor(images=[generated, target], return_tensors="pt")
                    pixels = inputs["pixel_values"].to(self._device)
                    features = self._clip.get_image_features(pixel_values=pixels)
                    values["i2i_clip_similarity"] = self._cosine(features[0], features[1])
                if self._siglip is not None:
                    inputs = self._siglip_processor(images=[generated, target], return_tensors="pt")
                    pixels = inputs["pixel_values"].to(self._device)
                    features = self._siglip.get_image_features(pixel_values=pixels)
                    values["i2i_siglip_similarity"] = self._cosine(features[0], features[1])
                if self._lpips_alex is not None:
                    from torchvision.transforms.functional import pil_to_tensor, resize

                    tensors = []
                    for image in (generated, target):
                        tensor = resize(pil_to_tensor(image).float() / 127.5 - 1.0, [256, 256])
                        tensors.append(tensor.unsqueeze(0).to(self._device))
                    if self._lpips_alex is not None:
                        values["i2i_lpips_alex"] = float(
                            self._lpips_alex(tensors[0], tensors[1]).item()
                        )
            if not values or not all(np.isfinite(value) for value in values.values()):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.i2i_dinov2_cls_similarity = values.get(
                "i2i_dinov2_cls_similarity"
            )
            sample.quality_metrics.i2i_dinov2_patch_similarity = values.get(
                "i2i_dinov2_patch_similarity"
            )
            sample.quality_metrics.i2i_clip_similarity = values.get("i2i_clip_similarity")
            sample.quality_metrics.i2i_siglip_similarity = values.get("i2i_siglip_similarity")
            sample.quality_metrics.i2i_lpips_alex = values.get("i2i_lpips_alex")
        except Exception as exc:
            logger.warning("i2i_learned failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        for name in (
            "_dinov2", "_clip", "_siglip", "_lpips_alex",
            "_dinov2_processor", "_clip_processor", "_siglip_processor",
        ):
            setattr(self, name, None)
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        super().on_dispose()
