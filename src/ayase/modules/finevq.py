"""FineVQ — fine-grained video quality assessment.

Runs the real FineVQ model from HuggingFace (``IntMeGroup/FineVQ_score``).
When the model cannot be loaded the metric is left unset — there is no
handcrafted stand-in for the named FineVQ score.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FineVQModule(PipelineModule):
    name = "finevq"
    description = "Fine-grained video quality (real FineVQ model)"
    default_config = {
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
        # Sub-dimension fusion weights consumed by the real model's post-processing
        # (kept for config compatibility; never mutated in place).
        "weights": {
            "sharpness": 0.20,
            "colorfulness": 0.15,
            "noise": 0.20,
            "temporal_stability": 0.25,
            "content_richness": 0.20,
        },
    }
    metric_groups = {
        "finevq_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Real FineVQ model from HuggingFace
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            model_name = "IntMeGroup/FineVQ_score"
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(model_name, trust_remote_code=trc, revision=rev).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trc, revision=rev)
            self._device = device
            self._backend = "finevq"
            self._ml_available = True
            logger.info("FineVQ loaded real model from HuggingFace on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("FineVQ unavailable: real FineVQ model could not be loaded (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "finevq":
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            self._process_real_model(sample)
        except Exception as e:
            logger.warning("FineVQ failed: %s", e)
        return sample

    def _process_real_model(self, sample: Sample) -> None:
        """Process using the real FineVQ model."""
        import torch
        import cv2
        from PIL import Image

        frames_pil = []
        frames_cv = self._load_frames(sample)
        if not frames_cv:
            return

        for f in frames_cv:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))

        try:
            inputs = self._processor(images=frames_pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            if hasattr(outputs, "logits"):
                score = outputs.logits.mean().item()
            elif isinstance(outputs, torch.Tensor):
                score = outputs.mean().item()
            elif isinstance(outputs, dict) and "score" in outputs:
                score = float(outputs["score"])
            else:
                score = None

            if score is not None:
                # Don't clamp: real FineVQ model may output scores outside [0,1]
                sample.quality_metrics.finevq_score = float(score)
        except Exception as e:
            logger.warning("FineVQ model inference failed: %s", e)

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
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
