"""VideoScore multi-dimensional video quality module (EMNLP 2024).

VideoScore (TIGER-Lab/VideoScore) is an Idefics2-based sequence-regression model:
a single forward pass returns 5 continuous dimension scores (1.0-4.0) —
visual quality, temporal consistency, dynamic degree, text-to-video alignment,
factual consistency — read directly from the regression head logits.

Inference follows the official model card: ``Idefics2ForSequenceClassification``
(from the ``mantis`` package shipped with VideoScore) with the regression query
prompt and one ``<image>`` token per sampled frame. Real-or-none: when the
model / ``mantis`` package is unavailable the module reports itself unavailable
and leaves the videoscore_* fields unset (no generate()/digit-parsing fallback).
"""

import logging
from typing import Optional

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Official VideoScore regression query prompt (5 dimensions, 1.0-4.0 each).
REGRESSION_QUERY_PROMPT = (
    "Suppose you are an expert in judging and evaluating the quality of "
    "AI-generated videos,\nplease watch the following frames of a given video "
    "and see the text prompt for generating the video,\nthen give scores from 5 "
    "different dimensions:\n"
    "(1) visual quality: the quality of the video in terms of clearness, "
    "resolution, brightness, and color\n"
    "(2) temporal consistency, both the consistency of objects or humans and the "
    "smoothness of motion or movements\n"
    "(3) dynamic degree, the degree of dynamic changes\n"
    "(4) text-to-video alignment, the alignment between the text prompt and the "
    "video content\n"
    "(5) factual consistency, the consistency of the video content with the "
    "common-sense and factual knowledge\n\n"
    "for each dimension, output a float number from 1.0 to 4.0,\n"
    "the higher the number is, the better the video performs in that sub-score,\n"
    "the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is "
    "like a real video)\n"
    "Here is an output example:\n"
    "visual quality: 3.2\ntemporal consistency: 2.7\ndynamic degree: 4.0\n"
    "text-to-video alignment: 2.3\nfactual consistency: 1.8\n\n"
    'For this video, the text prompt is "{text_prompt}",\n'
    "all the frames of video are as follows:\n\n"
)

ROUND_DIGIT = 3


class VideoScoreModule(PipelineModule):
    name = "videoscore"
    description = "VideoScore 5-dimensional video quality assessment (1-4 scale)"
    default_config = {
        "model_name": "TIGER-Lab/VideoScore",
        "num_frames": 16,
        "trust_remote_code": True,
        "model_revision": None,
    }
    metric_groups = {
        "videoscore_alignment": "alignment",
        "videoscore_dynamic": "motion",
        "videoscore_factual": "alignment",
        "videoscore_temporal": "temporal",
        "videoscore_visual": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import AutoProcessor
            from mantis.models.idefics2 import Idefics2ForSequenceClassification

            from ayase.runtime import resolve_torch_device

            model_name = self.config.get("model_name", "TIGER-Lab/VideoScore")
            device = resolve_torch_device(self.config.get("device", "auto"))
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            )
            self._model = (
                Idefics2ForSequenceClassification.from_pretrained(
                    model_name, torch_dtype=dtype, trust_remote_code=trc, revision=rev
                )
                .eval()
                .to(device)
            )
            self._device = device
            self._ml_available = True
            self._backend = "videoscore"
            logger.info("VideoScore (Idefics2 regression) loaded on %s", device)
        except (ImportError, Exception) as e:
            self._backend = "unavailable"
            logger.warning(
                "VideoScore unavailable (needs the `mantis` package + weights): %s", e
            )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "videoscore":
            return sample

        try:
            scores = self._compute_scores(sample)
            if scores:
                sample.quality_metrics.videoscore_visual = scores.get("visual_quality")
                sample.quality_metrics.videoscore_temporal = scores.get("temporal_consistency")
                sample.quality_metrics.videoscore_dynamic = scores.get("dynamic_degree")
                sample.quality_metrics.videoscore_alignment = scores.get("text_video_alignment")
                sample.quality_metrics.videoscore_factual = scores.get("factual_consistency")
        except Exception as e:
            logger.warning("VideoScore processing failed: %s", e)
        return sample

    def _compute_scores(self, sample: Sample) -> Optional[dict]:
        import torch

        num_frames = self.config.get("num_frames", 16)
        frames = arrays_to_pil(sample_frames(sample.path, max_frames=num_frames, color="rgb"))
        if not frames:
            return None

        caption = sample.caption.text if sample.caption else "a video"
        eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=caption)
        # One <image> token per frame (per the official inference recipe).
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(frames):
            eval_prompt += "<image> " * (len(frames) - num_image_token)

        inputs = self._processor(text=eval_prompt, images=frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits  # [batch, 5]
        num_aspects = logits.shape[-1]
        aspect_scores = [
            round(float(logits[0, i].item()), ROUND_DIGIT) for i in range(num_aspects)
        ]
        dims = [
            "visual_quality",
            "temporal_consistency",
            "dynamic_degree",
            "text_video_alignment",
            "factual_consistency",
        ]
        return {dims[i]: aspect_scores[i] for i in range(min(len(dims), num_aspects))}
