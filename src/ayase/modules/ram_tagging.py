"""Recognize Anything Model (RAM++) tagging module.

Auto-tags video frames with multi-label tags from RAM++ (4,585 tag classes).
The HuggingFace repo `xinyu1205/recognize-anything-plus-model` only hosts
raw `.pth` checkpoints, not a transformers-compatible model — so we use the
official `recognize-anything` PyPI/GitHub package for loading and inference.

Installation:
    pip install git+https://github.com/xinyu1205/recognize-anything.git

The package brings in a vendored Swin transformer + BLIP-style image-tag
contrastive head. Image is preprocessed to 384x384 and run through the
model; `inference_ram()` returns space-separated EN tag string per image.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RAMTaggingModule(PipelineModule):
    name = "ram_tagging"
    description = "RAM++ multi-label tagging on sampled video frames"
    default_config = {
        "repo_id": "xinyu1205/recognize-anything-plus-model",
        "checkpoint_filename": "ram_plus_swin_large_14m.pth",
        "image_size": 384,
        "vit": "swin_l",
        "subsample": 4,
    }
    metric_groups = {
        "ram_tags": "scene",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import torch  # noqa: F401
            from huggingface_hub import hf_hub_download
            from ram.models import ram_plus
            from ram import get_transform
        except ImportError as e:
            logger.warning(
                "RAM tagging unavailable: %s. "
                "Install with: pip install git+https://github.com/xinyu1205/recognize-anything.git",
                e,
            )
            return

        try:
            ckpt = hf_hub_download(
                repo_id=self.config.get("repo_id"),
                filename=self.config.get("checkpoint_filename"),
            )
            image_size = int(self.config.get("image_size", 384))
            vit = self.config.get("vit", "swin_l")
            from ayase.runtime import resolve_torch_device
            device = resolve_torch_device(self.config.get("device", "auto"))

            model = ram_plus(pretrained=ckpt, image_size=image_size, vit=vit)
            model.eval()
            self._model = model.to(device)
            self._transform = get_transform(image_size=image_size)
            self._device = device
            self._backend = "ram_plus"
            self._ml_available = True
            logger.info("RAM++ model loaded on %s (image_size=%d, vit=%s)",
                        device, image_size, vit)
        except Exception as e:
            logger.warning("RAM tagging setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import numpy as np
            import torch
            from PIL import Image
            from ram import inference_ram

            frames = self._load_frames(sample)
            if not frames:
                return sample

            all_tags = set()
            for frame in frames:
                pil_img = Image.fromarray(np.ascontiguousarray(frame))
                img = self._transform(pil_img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    tags_en, _ = inference_ram(img, self._model)
                # tags_en is "tag1 | tag2 | tag3 ..."
                for t in (tags_en or "").split("|"):
                    t = t.strip()
                    if t:
                        all_tags.add(t)

            if all_tags:
                sample.quality_metrics.ram_tags = ", ".join(sorted(all_tags))
        except Exception as e:
            logger.warning("RAM tagging failed for %s: %s", sample.path, e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        from ayase.image import sample_frames

        subsample = self.config.get("subsample", 4)
        return list(sample_frames(sample.path, max_frames=subsample, color="rgb"))
