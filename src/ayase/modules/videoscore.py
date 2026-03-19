"""VideoScore multi-dimensional video quality module.

5 dimensions: visual quality, temporal consistency, dynamic degree,
text-video alignment, factual consistency. EMNLP 2024.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoScoreModule(PipelineModule):
    name = "videoscore"
    description = "VideoScore 5-dimensional video quality assessment (1-4 scale)"
    default_config = {
        "model_name": "TIGER-Lab/VideoScore",
        "num_frames": 8,
        "trust_remote_code": True,
        "model_revision": None,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._processor = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_name = self.config.get("model_name", "TIGER-Lab/VideoScore")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trc, revision=rev)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=trc,
                revision=rev,
            ).to(device)
            self._model.eval()
            self._device = device
            self._ml_available = True
            logger.info("VideoScore model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("VideoScore unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
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
        import cv2
        import torch
        from PIL import Image

        num_frames = self.config.get("num_frames", 8)

        # Load frames
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
        else:
            img = Image.open(str(sample.path)).convert("RGB")
            frames = [img]

        if not frames:
            return None

        caption = sample.caption.text if sample.caption else "a video"
        dimensions = [
            "visual_quality",
            "temporal_consistency",
            "dynamic_degree",
            "text_video_alignment",
            "factual_consistency",
        ]

        scores = {}
        for dim in dimensions:
            prompt = self._build_prompt(dim, caption)
            try:
                inputs = self._processor(
                    text=prompt, images=frames, return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs, max_new_tokens=16, do_sample=False
                    )

                response = self._processor.decode(outputs[0], skip_special_tokens=True)
                score = self._extract_score(response)
                if score is not None:
                    scores[dim] = score
            except Exception as e:
                logger.debug("VideoScore dim %s failed: %s", dim, e)

        return scores if scores else None

    def _build_prompt(self, dimension: str, caption: str) -> str:
        dim_prompts = {
            "visual_quality": "Rate the visual quality (clarity, resolution, color) of this video on a scale of 1-4.",
            "temporal_consistency": "Rate the temporal consistency (smoothness, coherence) of this video on a scale of 1-4.",
            "dynamic_degree": "Rate the dynamic degree (amount of meaningful motion) of this video on a scale of 1-4.",
            "text_video_alignment": f"Rate how well this video matches the description '{caption}' on a scale of 1-4.",
            "factual_consistency": "Rate the factual consistency (adherence to real-world common sense) of this video on a scale of 1-4.",
        }
        return dim_prompts.get(dimension, "Rate the quality of this video on a scale of 1-4.")

    def _extract_score(self, response: str) -> Optional[float]:
        """Extract numeric score from model response."""
        import re

        numbers = re.findall(r"(\d+\.?\d*)", response.split(":")[-1] if ":" in response else response)
        for num_str in numbers:
            val = float(num_str)
            if 0.0 <= val <= 5.0:
                return val
        return None
