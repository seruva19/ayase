"""Score per-sample visual agreement with ``sample.caption.text`` using VQAScore.

The vendored ``t2v_metrics`` backend asks whether the image shows the caption
and scores the probability of the affirmative answer. Images are scored once;
videos are sampled to at most four frames by default and their scores are
averaged. ``vqa_score_alignment`` is expected in [0, 1] and is better when
higher. The default model is ``clip-flant5-xxl``. CLIP-FlanT5 supports
``video_frame_input="pil"`` for decoded RGB frames; the default is JPEG quality 95
because the two inputs can produce different scores. Language/domain applicability
follows that model. This is not a reference-image or dataset-level metric, and
no CLIP/heuristic substitute is emitted when the backend or caption is absent.
Method and model source: https://github.com/linzhiqiu/t2v_metrics
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VQAScoreModule(PipelineModule):
    name = "vqa_score"
    implementation_status = "implemented"
    implementation_notes = "Vendored official t2v_metrics VQAScore model, actual caption/image score and mean selected video frames; field should be adapted for image-to-video extension."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    provenance = {"*": "published", "vqa_score_alignment": "adapted"}
    sources = {
        "vqa_score_alignment": "VQAScore, Lin et al. ECCV 2024; t2v_metrics — https://github.com/linzhiqiu/t2v_metrics",
    }
    description = "VQAScore text-visual alignment via VQA probability (0-1, higher=better)"
    default_config = {
        "model": "clip-flant5-xxl",
        "subsample": 4,
        "video_frame_input": "jpeg",
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
        """Score an image path or sampled video frames with t2v_metrics."""
        from contextlib import nullcontext
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image

        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        if sample.is_video:
            import cv2

            frame_input = self.config.get("video_frame_input", "jpeg")
            if frame_input == "pil":
                backend = getattr(self._model, "model", None)
                if not getattr(backend, "supports_pil_images", False):
                    raise ValueError("The selected VQAScore backend does not support PIL input")
                frame_directory = nullcontext(None)
            elif frame_input == "jpeg":
                frame_directory = TemporaryDirectory(prefix="ayase-vqascore-")
            else:
                raise ValueError("video_frame_input must be 'jpeg' or 'pil'")

            scores = []
            with frame_directory as directory:
                cap = cv2.VideoCapture(str(sample.path))
                try:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = list(range(0, total, max(1, total // subsample)))[:subsample]
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb)
                        if directory is not None:
                            frame_path = Path(directory) / f"{idx}.jpg"
                            image.save(frame_path, "JPEG", quality=95)
                            frame_source = str(frame_path)
                        else:
                            frame_source = image
                        scores.append(self._model(images=[frame_source], texts=[text]).item())
                finally:
                    cap.release()
            return float(np.mean(scores)) if scores else None

        return self._model(images=[str(sample.path)], texts=[text]).item()
    deviations = {"vqa_score_alignment": "Official VQAScore image quantity is averaged across selected video frames into the same field; this is the same image-to-video extension marked adapted for peer image metrics."}
