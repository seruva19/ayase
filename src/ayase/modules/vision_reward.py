"""VisionReward — fine-grained, QA-decomposed human preference reward.

AAAI 2026 (zai-org / THUDM). VisionReward decomposes human preference for a
generated image/video into a hierarchy of binary yes/no *judgment questions*.
A multimodal model (CogVLM2-Video for video, CogVLM2 for image) answers each
question; every answer is mapped to ``+1`` (yes) / ``-1`` (no) and combined
with a fixed per-question linear weight into a single interpretable score
(higher = better).

Reference:
    * Paper: "VisionReward: Fine-Grained Multi-Dimensional Human Preference
      Learning for Image and Video Generation" (AAAI 2026).
    * Code:  https://github.com/zai-org/VisionReward
    * Checkpoints: ``THUDM/VisionReward-Video`` (a.k.a. ``zai-org/VisionReward-Video``,
      base ``cogvlm2-video-llama3-chat``) and ``THUDM/VisionReward-Image``
      (base ``cogvlm2-llama3-chat-19B``).

Backend policy (STRICT, project-wide): the metric is produced **only** when the
real VisionReward checkpoint loads. There is no heuristic / CLIP-proxy
fallback — when the checkpoint or its dependencies are missing the metric is
left ``None`` and ``self._backend == "unavailable"``.

Scoring scheme (verbatim from ``inference-video.py`` in the upstream repo)::

    queries = [q.replace('[[prompt]]', prompt) for q in questions]   # 29 select Qs
    answers = [1 if model_answer == 'yes' else -1 for ...]           # per question
    score   = np.mean(answers * weight)                              # weight.json

The 29 ``VisionReward_video_qa_select.txt`` questions and the matching
``VisionReward_Video/weight.json`` array are embedded below verbatim so scoring
is deterministic and requires no extra downloads beyond the model checkpoint.

Images: the upstream image checkpoint is a SwissArmyTransformer (``sat``)
checkpoint whose scorer also depends on a separate ``VQAScore`` alignment model
— it is not loadable through Hugging Face ``transformers`` and pulling in a CLIP
alignment head would violate the no-proxy policy. Per the module contract,
images are therefore scored by routing the single frame through the video
checkpoint (a 1-frame "video"), which is the CogVLM2 model this module loads.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# --- Judgment questions (VisionReward_Video/VisionReward_video_qa_select.txt) ---
# Verbatim from github.com/zai-org/VisionReward. "[[prompt]]" is replaced with
# the sample caption at scoring time. Order is load-bearing: it is paired 1:1
# with _VIDEO_WEIGHTS below.
_VIDEO_QUESTIONS: Tuple[str, ...] = (
    'Does the video meet all the requirements stated in the text "[[prompt]]"?',
    'Does the video meet most of the requirements stated in the text "[[prompt]]"?',
    'Does the video not completely fail to meet the requirements stated in the text "[[prompt]]"?',
    'Is the composition aesthetically pleasing?',
    'Does the composition have no obvious flaws?',
    'Does the camera movement have no obvious flaws?',
    'Are the colors not significantly unattractive?',
    'Is the lighting perfectly accurate?',
    'Does the lighting have no obvious errors?',
    'Is there any lighting present?',
    'Is the lighting exceptionally beautiful?',
    'Is the lighting beautiful?',
    'Is the lighting not unattractive?',
    'Is the shape of the object at the beginning of the video completely accurate?',
    'Does the shape of the object at the beginning have no obvious errors?',
    'Is the shape of the object at the beginning not chaotic?',
    'Is the shape of the object perfectly maintained throughout the video?',
    'Is the shape of the object not chaotic throughout the video?',
    'Is the camera motion highly dynamic?',
    'Is the camera motion not minimal?',
    "Is the smoothness of the object's movement very good?",
    "Is the object's movement completely realistic?",
    'Is the image quality very stable?',
    'Are the details very refined?',
    'Are the details not rough?',
    'Are the details not significantly rough?',
    'Are all the letters correct?',
    'Are there any letters present?',
    'Is the video content part of the physical world?',
)

# --- Linear weights (VisionReward_Video/weight.json) ---
# One float per question, same order as _VIDEO_QUESTIONS.
_VIDEO_WEIGHTS: Tuple[float, ...] = (
    0.9543901856422174, 0.25239747290239256, 1.141818673357406, 0.03495652038170829,
    0.025237463294006605, 0.12600844108184325, 0.03221505988621183, 0.16286819641189937,
    0.21673935360893115, 0.01970324496671629, 0.13604019362894557, 0.09647134683834487,
    0.15490927135496332, 0.1294164598219855, 0.09891696198970226, 0.18839328668539077,
    0.1844335421380767, 0.2635526157239052, 0.11168980468489233, 0.05173789659242723,
    0.02562797122879315, 0.4389890596048526, 0.26857694964769424, 0.42925171836383774,
    0.00846154228462919, 0.12757277121689847, 0.05798205026065391, 0.1446334304609205,
    0.39418111694677266,
)

_PROMPT_PLACEHOLDER = "[[prompt]]"


class VisionRewardModule(PipelineModule):
    """Fine-grained QA-decomposed preference reward (VisionReward, AAAI 2026)."""

    name = "vision_reward"
    description = (
        "VisionReward fine-grained QA-decomposed human preference reward "
        "(CogVLM2-Video judgment questions, linearly weighted) — AAAI 2026"
    )
    default_config = {
        "device": "auto",
        "max_frames": 24,  # upstream load_video() samples 24 frames ('chat' strategy)
        "checkpoint": "THUDM/VisionReward-Video",  # CogVLM2-Video reward model
        "image_checkpoint": "THUDM/VisionReward-Image",  # sat-only; see module docstring
        "trust_remote_code": True,
        "model_revision": None,
        "temperature": 0.1,
        "max_new_tokens": 8,  # only the yes/no answer token is needed
        # Optional overrides for the embedded VisionReward question set / weights.
        # Both must be provided together and be equal length; otherwise the
        # embedded VisionReward_video_qa_select.txt + weight.json are used.
        "judgment_questions": None,
        "judgment_weights": None,
        "prompt_placeholder": _PROMPT_PLACEHOLDER,
    }
    metric_groups = {
        "vision_reward_score": "alignment",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._torch_dtype = None
        self._questions: Tuple[str, ...] = _VIDEO_QUESTIONS
        self._weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if self.test_mode:
            return

        checkpoint = self.config.get("checkpoint", "THUDM/VisionReward-Video")
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from ayase.runtime import resolve_torch_device, resolve_torch_dtype
        except ImportError as exc:
            self._backend = "unavailable"
            logger.info(
                "VisionReward unavailable: missing dependency (%s). Install "
                "torch + transformers and the '%s' checkpoint to enable it.",
                exc,
                checkpoint,
            )
            return

        try:
            self._questions, self._weights = self._resolve_questions_and_weights()

            device = resolve_torch_device(self.config.get("device", "auto"))
            trust_remote_code = self.config.get("trust_remote_code", True)
            revision = self.config.get("model_revision", None)
            dtype = resolve_torch_dtype(device, self.config.get("dtype", "auto"))

            self._tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "revision": revision,
            }
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            self._model = (
                AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
                .eval()
                .to(device)
            )

            self._device = device
            self._torch_dtype = dtype if dtype is not None else self._default_dtype()
            self._ml_available = True
            self._backend = f"hf:{checkpoint}"
            logger.info("VisionReward loaded %s on %s", checkpoint, device)
        except Exception as exc:  # noqa: BLE001 — surface as unavailable, no fallback
            self._backend = "unavailable"
            self._ml_available = False
            logger.info(
                "VisionReward unavailable: could not load checkpoint '%s' (%s).",
                checkpoint,
                exc,
            )

    def _resolve_questions_and_weights(self) -> Tuple[Tuple[str, ...], np.ndarray]:
        """Return the (questions, weights) pair, honoring config overrides."""
        questions = self.config.get("judgment_questions")
        weights = self.config.get("judgment_weights")
        if questions and weights is not None:
            questions = tuple(str(q) for q in questions)
            weights_arr = np.asarray(weights, dtype=np.float64)
            if len(questions) != weights_arr.shape[0]:
                raise ValueError(
                    "judgment_questions and judgment_weights must be equal length "
                    f"(got {len(questions)} vs {weights_arr.shape[0]})"
                )
            return questions, weights_arr
        return _VIDEO_QUESTIONS, np.asarray(_VIDEO_WEIGHTS, dtype=np.float64)

    def _default_dtype(self):
        import torch

        return torch.float32

    # ---------------------------------------------------------------- process
    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._model is None:
            return sample

        try:
            score = self._score_sample(sample)
            if score is not None:
                sample.quality_metrics.vision_reward_score = float(score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VisionReward scoring failed for %s: %s", sample.path, exc)
        return sample

    def _score_sample(self, sample: Sample) -> Optional[float]:
        import torch

        video = self._build_video_tensor(sample)
        if video is None:
            return None

        prompt = ""
        if getattr(sample, "caption", None) is not None:
            prompt = getattr(sample.caption, "text", "") or ""

        placeholder = self.config.get("prompt_placeholder", _PROMPT_PLACEHOLDER)
        weights = (
            self._weights
            if self._weights is not None
            else np.asarray(_VIDEO_WEIGHTS, dtype=np.float64)
        )

        answers: List[int] = []
        with torch.inference_mode():
            for question in self._questions:
                query = question.replace(placeholder, prompt)
                answers.append(self._answer_yes_no(video, query))

        answer_arr = np.asarray(answers, dtype=np.float64)
        return float(np.mean(answer_arr * weights))

    def _build_video_tensor(self, sample: Sample):
        """Sample RGB frames and lay them out as CogVLM2-Video expects (C, T, H, W)."""
        import torch

        max_frames = int(self.config.get("max_frames", 24))
        frames = sample_frames(sample.path, max_frames=max_frames, color="rgb")
        if not frames:
            return None

        # sample_frames returns READ-ONLY views; copy into one contiguous block.
        stacked = np.ascontiguousarray(np.stack([np.asarray(f) for f in frames]))
        # (T, H, W, C) uint8 -> (C, T, H, W), matching upstream video_data.permute(3,0,1,2).
        video = torch.from_numpy(stacked).permute(3, 0, 1, 2)
        return video

    def _answer_yes_no(self, video, query: str) -> int:
        """Run one judgment question through the model; +1 for 'yes', -1 otherwise."""
        import torch

        inputs = self._model.build_conversation_input_ids(
            tokenizer=self._tokenizer,
            query=query,
            images=[video],
            history=[],
            template_version="chat",
        )
        image = inputs["images"][0].to(self._device)
        if self._torch_dtype is not None:
            image = image.to(self._torch_dtype)
        model_inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(self._device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(self._device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(self._device),
            "images": [[image]],
        }
        gen_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 8)),
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": float(self.config.get("temperature", 0.1)),
        }
        outputs = self._model.generate(**model_inputs, **gen_kwargs)
        outputs = outputs[:, model_inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return 1 if text.strip().lower().startswith("yes") else -1
