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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._transform = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ram_plus(pretrained=ckpt, image_size=image_size, vit=vit)
            model.eval()
            self._model = model.to(device)
            self._transform = get_transform(image_size=image_size)
            self._device = device
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
            import cv2
            import torch
            from PIL import Image
            from ram import inference_ram

            frames = self._load_frames(sample)
            if not frames:
                return sample

            all_tags = set()
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
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
        import cv2

        subsample = self.config.get("subsample", 4)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames
