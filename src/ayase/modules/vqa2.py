"""VQA² --- Visual Question Answering for Video Quality Assessment (ACM MM 2025).

Paper:  https://arxiv.org/abs/2411.03795
GitHub: https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment
Model:  https://huggingface.co/q-future/VQA-UGC-Scorer-llava_qwen  (7B, BF16)

VQA² is a LLaVA-OneVision / Qwen2 large multimodal model fine-tuned for
no-reference visual quality scoring. A frame (image) or a slow/fast pair of
frame streams (video) is fed to the model with a fixed quality prompt; the
score is read from the next-token logits over five quality-level tokens
(the Q-Align "discrete text-defined levels" trick), not parsed from free text.

Exact published inference protocol (from the repo's eval scripts
``quality_scoring/llava/eval/model_score_image.py`` and
``model_score_UGC_video.py``):

  * Model is loaded via the repo's *own bundled* ``llava`` package
    (``from llava.model.builder import load_pretrained_model``) — the
    checkpoint's ``model_type`` is ``llava_qwen`` / arch
    ``LlavaQwenForCausalLM`` with a custom "slowfast" video forward, so it is
    NOT loadable by stock ``transformers`` (no ``auto_map``, no registered
    ``llava_qwen`` config).
  * Prompt is built with a Qwen chat template (``preprocess_qwen``); the
    assistant turn is primed with ``"The overall quality of the image is"``
    for images and left empty for the slow/fast video forward.
  * ``logits = model(input_ids, images=...)["logits"][:, -3]`` — the third-
    from-last position predicts the quality-level word.
  * Five fixed token ids are read: 1550, 1661, 6624, 7852, 3347
    (excellent/high, good, fair, poor, bad/low).
  * ``wa5``: softmax over those five logits, then a weighted average with
    weights ``[1.0, 0.8, 0.6, 0.4, 0.2]`` → a score in ``[0.2, 1.0]``.

vqa2_score --- higher = better (0.2-1.0 range from wa5)

REVIVAL NOTES
=============
Metric:        vqa2  (VQA², ACM MM 2025, org q-future)
Category:      NR video/image quality (LMM, Q-Align-style level scoring)
Field:         QualityMetrics.vqa2_score
Why provisional:
    The released checkpoint ``q-future/VQA-UGC-Scorer-llava_qwen`` declares
    ``model_type: "llava_qwen"`` / ``architectures: ["LlavaQwenForCausalLM"]``
    with NO ``auto_map`` and NO bundled modeling file, so ``transformers``
    (any version) cannot instantiate it. Running it requires the repo's own
    *modified* ``llava`` package (LLaVA-OneVision fork with custom
    ``slowfast`` / ``slowfast_projector`` modules) — a heavy, non-turnkey
    framework that pins an older transformers and (typically) flash-attn.
    It is not installable as a standard pip dependency alongside the rest of
    Ayase, so it is left graceful-unavailable rather than substituting a
    different quality model for the published metric.
To revive:
    1. git clone https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment
    2. cd quality_scoring && pip install -e .   (installs the bundled `llava`
       package + its pinned deps; a compatible flash-attn helps but the code
       runs with eager attention too)
    3. Download q-future/VQA-UGC-Scorer-llava_qwen (~16 GB).
    4. Set ``model_path`` in this module's config and enable it. ``setup()``
       will import ``llava.model.builder.load_pretrained_model`` and
       ``process()`` will run the exact protocol above (already wired below).
    5. Verify a finite score on a real clip, then flip ``provisional = False``
       and ``_backend = "vqa2_llava"`` permanently.
Source:
    Repo eval scripts (verbatim): quality_scoring/llava/eval/model_score_image.py,
    quality_scoring/llava/eval/model_score_UGC_video.py  (wa5 + token ids +
    logits[:, -3]); model card config.json (llava_qwen arch, siglip tower).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Five quality-level token ids read from the next-token logits (identical in
# both the image and video eval scripts). Order: excellent/high, good, fair,
# poor, bad/low. wa5 weights map them onto a 0.2-1.0 score.
VQA2_QUALITY_TOKEN_IDS = [1550, 1661, 6624, 7852, 3347]
VQA2_WA5_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]
# Position of the logits used for scoring (third-from-last token).
VQA2_LOGIT_POSITION = -3
# Assistant prime for the single-image path (video primes with "").
VQA2_IMAGE_ASSISTANT_PREFIX = "The overall quality of the image is"


def _wa5(logits5: List[float]) -> float:
    """Softmax over the five quality logits, weighted-average to [0.2, 1.0].

    Mirrors ``wa5`` in the VQA² eval scripts exactly.
    """
    logprobs = np.asarray(logits5, dtype=np.float64)
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return float(np.inner(probs, np.asarray(VQA2_WA5_WEIGHTS, dtype=np.float64)))


class VQA2Module(PipelineModule):
    name = "vqa2"
    # Real backend requires the repo's custom `llava` package (not a standard
    # install) — see REVIVAL NOTES in the module docstring. Kept provisional
    # until a finite score is verified end-to-end with that package present.
    provisional = True
    description = "VQA^2 LMM video/image quality assessment (ACM MM 2025)"
    default_config = {
        "model_path": "q-future/VQA-UGC-Scorer-llava_qwen",
        "model_base": None,
        "device": "auto",
        "subsample": 8,      # every Nth frame for the "fast" video stream
        "max_frames": 32,    # cap frames fed to the model (VRAM safety)
    }
    metric_groups = {
        "vqa2_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "q-future/VQA-UGC-Scorer-llava_qwen")
        self.model_base = self.config.get("model_base", None)
        self.device_config = self.config.get("device", "auto")
        self.subsample = self.config.get("subsample", 8)
        self.max_frames = self.config.get("max_frames", 32)

        self.device = None
        self._ml_available = False
        self._backend = "unavailable"
        self._tokenizer = None
        self._model = None
        self._image_processor = None

    # ------------------------------------------------------------------
    # Setup — load the released VQA² LLaVA-Qwen checkpoint via the repo's
    # bundled `llava` package. Real backend or None; no substitute model.
    # ------------------------------------------------------------------
    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch  # noqa: F401
            from llava.model.builder import load_pretrained_model  # type: ignore
            from llava.mm_utils import get_model_name_from_path  # type: ignore
            from llava.utils import disable_torch_init  # type: ignore
            from ayase.runtime import resolve_torch_device

            self.device = resolve_torch_device(self.device_config)

            disable_torch_init()
            model_name = get_model_name_from_path(self.model_path)
            logger.info("Loading VQA^2 checkpoint %s (~16 GB)...", self.model_path)
            tokenizer, model, image_processor, _ctx = load_pretrained_model(
                self.model_path,
                self.model_base,
                model_name,
                attn_implementation=None,
            )
            model.half()
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._image_processor = image_processor
            self._ml_available = True
            self._backend = "vqa2_llava"
            logger.info("VQA^2 initialised (bundled llava backend)")
            return
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover - depends on external package
            logger.warning("VQA^2 setup failed: %s", e)
            self._ml_available = False
            self._backend = "unavailable"
            return

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "VQA^2 unavailable: the released VQA^2 `llava` package is not "
            "installed (see REVIVAL NOTES in vqa2.py). Metric left as None."
        )

    # ------------------------------------------------------------------
    # Qwen chat-template builder — faithful port of `preprocess_qwen` from
    # the VQA² eval scripts, for a single (user, assistant) turn.
    # ------------------------------------------------------------------
    def _build_input_ids(self, user_value: str, assistant_value: str):
        import re
        import torch
        from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore

        tokenizer = self._tokenizer
        im_start, im_end = tokenizer.additional_special_tokens_ids
        nl_tokens = tokenizer("\n").input_ids
        _system = tokenizer("system").input_ids + nl_tokens
        system_message = "You are a helpful assistant."

        input_id: List[int] = []
        input_id += [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens

        source = [user_value, {"from": "gpt", "value": assistant_value}]
        for j, sentence in enumerate(source):
            role = "<|im_start|>user" if j == 0 else "<|im_start|>assistant"
            if j == 0 and DEFAULT_IMAGE_TOKEN in user_value:
                texts = user_value.split(DEFAULT_IMAGE_TOKEN)
                _input_id = tokenizer(role).input_ids + nl_tokens
                for i, text in enumerate(texts):
                    _input_id += tokenizer(text).input_ids
                    if i < len(texts) - 1:
                        _input_id += [IMAGE_TOKEN_INDEX]
                _input_id += [im_end] + nl_tokens
            else:
                value = sentence["value"] if isinstance(sentence, dict) else sentence
                if value is None:
                    _input_id = tokenizer(role).input_ids + nl_tokens
                else:
                    _input_id = (
                        tokenizer(role).input_ids + nl_tokens
                        + tokenizer(value).input_ids + [im_end] + nl_tokens
                    )
            input_id += _input_id

        return torch.tensor([input_id], dtype=torch.long)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_from_logits(self, logits_row) -> float:
        vals = [logits_row[tid].item() for tid in VQA2_QUALITY_TOKEN_IDS]
        return _wa5(vals)

    def _score_image(self, pil_image) -> Optional[float]:
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore

        proc = self._image_processor.preprocess(pil_image, return_tensors="pt")
        pixel_values = proc["pixel_values"]
        image_sizes = proc.get("image_sizes", None)

        input_ids = self._build_input_ids(
            DEFAULT_IMAGE_TOKEN, VQA2_IMAGE_ASSISTANT_PREFIX
        ).to(self.device)
        image_tensors = [pixel_values.half().to(self.device)]

        kwargs = {"images": image_tensors, "modalities": ["image"]}
        if image_sizes is not None:
            kwargs["image_sizes"] = image_sizes

        with torch.no_grad():
            out = self._model(input_ids, **kwargs)
        logits = out["logits"][:, VQA2_LOGIT_POSITION]
        return self._score_from_logits(logits.mean(0).float())

    def _score_video(self, frames: List["object"]) -> Optional[float]:
        """Slow/fast dual-stream forward (VQA² custom video path).

        slow = all sampled frames (truncated to a multiple of 4);
        fast = ~1 fps subset. Both are packed as ``images=[[slow],[fast]]``.
        """
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore

        if not frames:
            return None

        slow_frames = frames[: len(frames) // 4 * 4] or frames
        fast_frames = frames[:: max(1, self.subsample)] or frames[:1]

        slow_t = self._image_processor.preprocess(slow_frames, return_tensors="pt")["pixel_values"]
        fast_t = self._image_processor.preprocess(fast_frames, return_tensors="pt")["pixel_values"]

        image_tensors = [
            [slow_t.half().to(self.device)],
            [fast_t.half().to(self.device)],
        ]
        # Two image tokens (slow + fast); empty assistant turn, as in the
        # video eval script.
        input_ids = self._build_input_ids(
            DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN, ""
        ).to(self.device)

        with torch.no_grad():
            out = self._model(input_ids, images=image_tensors)
        logits = out["logits"][:, VQA2_LOGIT_POSITION]
        return self._score_from_logits(logits.mean(0).float())

    # ------------------------------------------------------------------
    # Frame extraction (OpenCV; decord not required)
    # ------------------------------------------------------------------
    def _load_frames(self, path) -> List["object"]:
        import cv2
        from PIL import Image

        frames: List["object"] = []
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return frames
        try:
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        finally:
            cap.release()
        return frames

    def _load_image(self, path):
        from PIL import Image

        try:
            return Image.open(str(path)).convert("RGB")
        except Exception:
            return None

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "vqa2_llava":
            return sample

        try:
            if sample.is_video:
                frames = self._load_frames(sample.path)
                score = self._score_video(frames)
            else:
                img = self._load_image(sample.path)
                score = self._score_image(img) if img is not None else None

            if score is not None and np.isfinite(score):
                sample.quality_metrics.vqa2_score = float(score)
        except Exception as e:
            logger.warning("VQA^2 failed for %s: %s", sample.path, e)
        return sample
