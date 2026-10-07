"""VLM Likert physical-commonsense / semantic-adherence probing — own metric
inspired by VideoPhy-2; published protocol not reproduced.
evaluation for T2V (Bansal et al., arXiv:2406.03520 / arXiv:2503.06800).

Complements the existing trajectory-based ``physics`` module: that one
measures motion plausibility from optical-flow tracks; this one asks a
VLM whether the rendered video respects everyday physics and matches
the prompt's intent.

Outputs:
    * ``vlm_pc_likert``  — physical commonsense, 0..1
    * ``vlm_sa_likert``  — semantic adherence to caption, 0..1

Backend: LLaVA-NeXT-Video with a Likert prompt template inspired by
VideoPhy-2. If the VLM cannot be loaded the module emits no scores — a
binary judge on a single frame or a trajectory-based physics score are
different quantities and are not substituted under these fields.
"""

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


PHYSICS_PROMPT = (
    "You are evaluating a video for physical plausibility. "
    "Does the motion in this video obey everyday physics (gravity, "
    "rigid-body contact, momentum, fluid behavior)? "
    "Respond with a single integer 1-5 where 5 means fully plausible "
    "and 1 means obvious physics violations."
)
SEMANTIC_PROMPT = (
    "Does the video faithfully depict the caption: '{caption}'? "
    "Respond with a single integer 1-5 where 5 means fully matches "
    "and 1 means largely unrelated."
)


class VideoPhyModule(PipelineModule):
    name = "vlm_phy"
    provenance = "own"
    sources = {
        "vlm_pc_likert": "VideoPhy-2, Bansal et al. arXiv:2503.06800 — name only — https://github.com/Hritikbansal/videophy",
        "vlm_sa_likert": "VideoPhy-2, Bansal et al. arXiv:2503.06800 — name only — https://github.com/Hritikbansal/videophy",
    }
    description = "VLM Likert-scale physics-adherence probing (own, VideoPhy-2-inspired)"
    default_config = {
        "model_name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "num_frames": 8,
        "backend": "auto",  # "auto" | "vlm"
        "max_new_tokens": 8,
        "models_dir": "models",
    }
    metric_groups = {
        "vlm_pc_likert": "motion",
        "vlm_sa_likert": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "llava-hf/LLaVA-NeXT-Video-7B-hf")
        self.num_frames = self.config.get("num_frames", 8)
        self.backend_pref = self.config.get("backend", "auto")
        self.max_new_tokens = self.config.get("max_new_tokens", 8)
        self.models_dir = self.config.get("models_dir", "models")
        self._device = "cpu"
        self._vlm_model = None
        self._vlm_processor = None
        self.active_backend = "unavailable"

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            import torch
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except ImportError:
            return

        if self.backend_pref in ("auto", "vlm") and self._try_load_vlm():
            self.active_backend = "vlm"
        logger.info(f"VideoPhy initialized with backend={self.active_backend}")

    def _try_load_vlm(self) -> bool:
        try:
            import torch
            from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
            from ayase.runtime import from_pretrained_with_attention

            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32
            self._vlm_processor = LlavaNextVideoProcessor.from_pretrained(
                self.model_name, cache_dir=self.models_dir,
            )
            self._vlm_model = from_pretrained_with_attention(
                LlavaNextVideoForConditionalGeneration,
                self.model_name,
                self.config,
                device=self._device,
                cache_dir=self.models_dir,
                torch_dtype=dtype,
            ).to(self._device).eval()
            return True
        except Exception as e:
            logger.debug(f"VideoPhy: LLaVA-NeXT-Video unavailable: {e}")
            return False

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self.active_backend != "vlm":
            return sample
        caption = sample.caption.text if sample.caption else ""

        try:
            pc, sa = self._score_vlm(sample, caption)
        except Exception as e:
            logger.debug(f"VideoPhy scoring failed for {sample.path}: {e}")
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if pc is not None:
            sample.quality_metrics.vlm_pc_likert = round(float(pc), 3)
        if sa is not None:
            sample.quality_metrics.vlm_sa_likert = round(float(sa), 3)
        return sample

    def _score_vlm(self, sample: Sample, caption: str) -> Tuple[Optional[float], Optional[float]]:
        frames = self._sample_frames(sample.path, self.num_frames)
        if frames is None:
            return None, None

        pc_response = self._ask_vlm(frames, PHYSICS_PROMPT)
        sa_response = self._ask_vlm(frames, SEMANTIC_PROMPT.format(caption=caption or "the scene"))
        return _parse_likert(pc_response), _parse_likert(sa_response)

    def _ask_vlm(self, frames: np.ndarray, prompt: str) -> str:
        import torch

        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},
        ]}]
        text = self._vlm_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._vlm_processor(
            text=[text], videos=[list(frames)], return_tensors="pt", padding=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._vlm_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self._vlm_processor.batch_decode(out, skip_special_tokens=True)[0]

    def _sample_frames(self, path, n: int) -> Optional[np.ndarray]:
        frames = sample_frames(path, max_frames=n, color="rgb")
        if len(frames) < n:
            return None
        return np.stack(frames, axis=0)


def _parse_likert(text: str) -> Optional[float]:
    # VideoPhy-2 prompts ask for a 1-5 integer; we accept the first digit.
    # If the response has no parseable rating, return None rather than a
    # fabricated neutral value.
    m = re.search(r"[1-5]", text)
    if not m:
        return None
    return (int(m.group(0)) - 1) / 4.0
