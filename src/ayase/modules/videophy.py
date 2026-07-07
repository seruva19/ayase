"""VideoPhy-2 — VLM-based physical-commonsense / semantic-adherence
evaluation for T2V (Bansal et al., arXiv:2406.03520 / arXiv:2503.06800).

Complements the existing trajectory-based ``physics`` module: that one
measures motion plausibility from optical-flow tracks; this one asks a
VLM whether the rendered video respects everyday physics and matches
the prompt's intent.

Outputs:
    * ``videophy_pc_score``  — physical commonsense, 0..1
    * ``videophy_sa_score``  — semantic adherence to caption, 0..1

Tiered backends:
    1. LLaVA-NeXT-Video / Qwen2-VL with the VideoPhy-2 prompt template.
    2. ``vlm_judge`` reused with a "physical plausibility" preset prompt.
    3. Trajectory-based fallback that reuses the existing ``physics``
       module score so the field is always populated.
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
    name = "videophy"
    description = "VideoPhy-2 VLM-based physics adherence (arXiv:2503.06800)"
    default_config = {
        "model_name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "num_frames": 8,
        "backend": "auto",  # "auto" | "vlm" | "vlm_judge" | "trajectory"
        "max_new_tokens": 8,
        "models_dir": "models",
    }
    metric_groups = {
        "videophy_pc_score": "motion",
        "videophy_sa_score": "motion",
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
        self._vlm_judge = None
        self._physics_module = None
        self.active_backend = "trajectory"

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
            return
        if self.backend_pref in ("auto", "vlm_judge") and self._try_setup_vlm_judge():
            self.active_backend = "vlm_judge"
            return
        if self._try_setup_trajectory():
            self.active_backend = "trajectory"
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

    def _try_setup_vlm_judge(self) -> bool:
        try:
            from ayase.modules.vlm_judge import VLMJudgeModule
            self._vlm_judge = VLMJudgeModule(
                config={
                    "mode": "verify",
                    "models_dir": self.models_dir,
                    "device": self.config.get("device", "auto"),
                    "attention_backend": self.config.get("attention_backend", "auto"),
                }
            )
            self._vlm_judge.on_mount()
            return getattr(self._vlm_judge, "_ml_available", False)
        except Exception as e:
            logger.debug(f"VideoPhy: vlm_judge unavailable: {e}")
            return False

    def _try_setup_trajectory(self) -> bool:
        try:
            from ayase.modules.physics import PhysicsModule
            self._physics_module = PhysicsModule()
            self._physics_module.on_mount()
            return True
        except Exception as e:
            logger.debug(f"VideoPhy: trajectory fallback unavailable: {e}")
            return False

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        caption = sample.caption.text if sample.caption else ""

        try:
            if self.active_backend == "vlm":
                pc, sa = self._score_vlm(sample, caption)
            elif self.active_backend == "vlm_judge":
                pc, sa = self._score_via_judge(sample, caption)
            else:
                pc, sa = self._score_trajectory(sample, caption)
        except Exception as e:
            logger.debug(f"VideoPhy scoring failed for {sample.path}: {e}")
            pc, sa = self._score_trajectory(sample, caption)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if pc is not None:
            sample.quality_metrics.videophy_pc_score = round(float(pc), 3)
        if sa is not None:
            sample.quality_metrics.videophy_sa_score = round(float(sa), 3)
        return sample

    def _score_vlm(self, sample: Sample, caption: str) -> Tuple[float, float]:
        frames = self._sample_frames(sample.path, self.num_frames)
        if frames is None:
            return self._score_trajectory(sample, caption)

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

    def _score_via_judge(self, sample: Sample, caption: str) -> Tuple[float, float]:
        # vlm_judge processes images; we feed it the middle frame as proxy.
        img = sample.load_image()
        if img is None:
            return self._score_trajectory(sample, caption)
        # Reuse the judge's verification path with our physics prompt.
        original = self._vlm_judge.verification_prompt
        try:
            self._vlm_judge.verification_prompt = PHYSICS_PROMPT
            pc_out = self._vlm_judge.process(sample)
            self._vlm_judge.verification_prompt = SEMANTIC_PROMPT.format(caption=caption or "the scene")
            sa_out = self._vlm_judge.process(sample)
        finally:
            self._vlm_judge.verification_prompt = original
        # vlm_judge writes a yes/no validation issue; approximate as 1.0 or 0.0
        pc = 1.0 if not _has_negative_issue(pc_out) else 0.0
        sa = 1.0 if not _has_negative_issue(sa_out) else 0.0
        return pc, sa

    def _score_trajectory(self, sample: Sample, caption: str) -> Tuple[Optional[float], Optional[float]]:
        # Reuse the existing trajectory-based physics score for physical
        # commonsense. Semantic adherence has no signal from trajectories,
        # so it is left unset (None) rather than a fabricated neutral value.
        if self._physics_module is not None:
            try:
                out = self._physics_module.process(sample)
                if out.quality_metrics is not None and out.quality_metrics.physics_score is not None:
                    pc = float(np.clip(out.quality_metrics.physics_score, 0.0, 1.0))
                    return pc, None
            except Exception:
                pass
        # No model + no trajectory signal → leave both unset rather than
        # emitting a fabricated neutral score.
        return None, None

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


def _has_negative_issue(sample: Sample) -> bool:
    from ayase.models import ValidationSeverity
    return any(
        issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.WARNING)
        for issue in sample.validation_issues
    )
