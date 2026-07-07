"""VQAScore text-visual alignment module.

Uses a VQA model to score "Does this figure show {text}?" probability.
ECCV 2024, outperforms CLIPScore on compositional text prompts.

This module requires the vendored ``t2v_metrics`` VQAScore model. When it is
unavailable the metric is left ``None`` — CLIP cosine similarity (i.e.
CLIPScore, the weaker method VQAScore is designed to beat) is not substituted
for the published metric.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VQAScoreModule(PipelineModule):
    name = "vqa_score"
    description = "VQAScore text-visual alignment via VQA probability (0-1, higher=better)"
    default_config = {
        "model": "clip-flant5-xxl",
        "subsample": 4,
    }
    models = [
        {
            "id": "clip-flant5-xxl",
            "type": "other",
            "task": "Vendored t2v_metrics VQAScore model",
            "install": "bundled in ayase.vendor.t2v_metrics",
        },
    ]
    metric_info = {
        "vqa_score_alignment": "VQAScore text-visual alignment probability (0-1, higher=better)",
    }
    metric_groups = {
        "vqa_score_alignment": "alignment",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        try:
            from ayase.vendor.t2v_metrics import VQAScore as VQAScoreMetric

            model_name = self.config.get("model", "clip-flant5-xxl")
            self._model = VQAScoreMetric(model=model_name)
            self._ml_available = True
            self._backend = "t2v_metrics"
            logger.info("VQAScore model loaded: %s", model_name)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("VQAScore unavailable: ayase.vendor.t2v_metrics not importable")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("VQAScore unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._model is None:
            return sample
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            score = self._compute_vqascore(sample)
            if score is not None:
                sample.quality_metrics.vqa_score_alignment = float(score)
        except Exception as e:
            logger.warning("VQAScore processing failed: %s", e)
        return sample

    def _compute_vqascore(self, sample: Sample) -> Optional[float]:
        """Compute VQAScore using t2v_metrics.

        The underlying ``t2v_metrics`` API expects ``images`` to be a list of
        path strings (it calls ``.startswith()`` on each entry to detect
        URLs). Passing PIL.Image objects raises
        ``'Image' object has no attribute 'startswith'``. So: for images we
        pass ``sample.path`` directly; for videos we dump each sampled frame
        to a temporary JPEG and pass the path.
        """
        import tempfile
        from PIL import Image

        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        if sample.is_video:
            import cv2

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            scores = []
            tmp_paths: List[str] = []
            try:
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    tf.close()
                    pil_img.save(tf.name, "JPEG", quality=95)
                    tmp_paths.append(tf.name)
                    score = self._model(images=[tf.name], texts=[text]).item()
                    scores.append(score)
            finally:
                cap.release()
                import os
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            return float(np.mean(scores)) if scores else None

        return self._model(images=[str(sample.path)], texts=[text]).item()
