"""CameraBench camera-motion taxonomy classification (arXiv 2504.15376).

Classifies the camera motion of a clip over the 15 official CameraBench
motion primitives (move/tilt/pan/roll/zoom/static) using the fine-tuned
Qwen2.5-VL model released by the CameraBench authors:

  * checkpoint: ``chancharikm/qwen2.5-vl-7b-cam-motion``
    (32B / 72B variants exist: ``chancharikm/qwen2.5-vl-32b-cam-motion``,
    ``chancharikm/qwen2.5-vl-72b-cam-motion``)
  * processor:  ``Qwen/Qwen2.5-VL-7B-Instruct``
  * project:    https://linzhiqiu.github.io/papers/camerabench/
  * code:       https://github.com/sy77777en/CameraBench

Real backend only — there is no optical-flow heuristic classifier tier. If the
VLM cannot be loaded the module sets ``_backend = "unavailable"`` and leaves both
outputs unset.

On success the argmax primitive is stored in
``sample.metadata["camera_motion_class"]``, its probability in
``quality_metrics.camera_motion_class_confidence``, and the full per-primitive
probability map in ``sample.metadata["camera_motion_primitives"]``.

Classification follows the benchmark's binary-classification protocol: each of
the 15 verbatim primitive questions (``camerabench/data/binary_classification``
in ``linzhiqiu/t2v_metrics``) is asked independently as ``"{question} Please
only answer Yes or No."`` and scored by the softmax probability of the ``Yes``
token in the next-token distribution.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# Official CameraBench binary-classification primitives: {label: verbatim
# question from the benchmark data (camerabench/data/binary_classification in
# linzhiqiu/t2v_metrics)}. Each is asked independently and scored as P("Yes").
CAMERA_MOTION_LABELS = {
    "move_down": "Does the camera move downward (not tilting down) with respect to the initial frame?",
    "move_in": "Does the camera move forward (not zooming in) with respect to the initial frame?",
    "move_left": "Does the camera move leftward in the scene?",
    "move_out": "Does the camera move backward (not zooming out) with respect to the initial frame?",
    "move_right": "Does the camera move rightward in the scene?",
    "move_up": "Does the camera move upward (not tilting up) with respect to the initial frame?",
    "pan_left": "Does the camera pan to the left?",
    "pan_right": "Does the camera pan to the right?",
    "roll_clockwise": "Does the camera roll clockwise?",
    "roll_counterclockwise": "Does the camera roll counterclockwise?",
    "static": "Is the camera completely still without any motion or shaking?",
    "tilt_down": "Does the camera tilt downward?",
    "tilt_up": "Does the camera tilt upward?",
    "zoom_in": "Does the camera zoom in?",
    "zoom_out": "Does the camera zoom out?",
}


def _store_camera_motion_class(sample: Sample, label: str) -> None:
    """Attach the predicted class to ``sample.metadata["camera_motion_class"]``."""
    sample.metadata["camera_motion_class"] = label


class CameraBenchModule(PipelineModule):
    name = "camerabench"
    provenance = "adapted"
    sources = {
        "camera_motion_class_confidence": "CameraBench (Lin et al. 2025), chancharikm/qwen2.5-vl-7b-cam-motion model — https://github.com/sy77777en/CameraBench",
    }
    deviations = {
        "camera_motion_class_confidence": "the official 15 per-primitive binary questions (verbatim); the field keeps argmax P(Yes) for compatibility, all primitive probabilities in metadata['camera_motion_primitives']",
    }
    description = (
        "CameraBench camera-motion taxonomy classification via the fine-tuned "
        "Qwen2.5-VL model (chancharikm/qwen2.5-vl-7b-cam-motion)"
    )
    default_config = {
        "model_id": "chancharikm/qwen2.5-vl-7b-cam-motion",
        "processor_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "num_frames": 16,
        "fps": 8.0,
    }
    metric_groups = {
        "camera_motion_class_confidence": "motion",
    }
    metric_info = {
        "camera_motion_class_confidence": (
            "Confidence (softmax P(yes)) of the predicted CameraBench camera-motion "
            "class stored in sample.metadata['camera_motion_class']"
        ),
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))
        try:
            model, processor = self._load_qwen()
            self._model = model
            self._processor = processor
            self._backend = "qwen2.5-vl-camerabench"
            self._ml_available = True
            logger.info(
                "camerabench: loaded %s on %s",
                self.config.get("model_id"),
                self._device,
            )
            return
        except ImportError as exc:
            logger.info("camerabench: Qwen2.5-VL unavailable (missing dependency): %s", exc)
        except Exception as exc:  # pragma: no cover - depends on optional weights
            logger.info("camerabench: Qwen2.5-VL load failed: %s", exc)

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "camerabench unavailable: the fine-tuned Qwen2.5-VL checkpoint could not "
            "be loaded; camera_motion_class / camera_motion_class_confidence will not "
            "be populated by this module."
        )

    def _load_qwen(self):
        """Load and cache the shared Qwen2.5-VL model + processor."""
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # qwen_vl_utils.process_vision_info is the canonical way to build the
        # Qwen2.5-VL video inputs (see the model card). Require it up front so a
        # missing dependency disables the module cleanly (setup -> unavailable)
        # rather than failing later inside process().
        from qwen_vl_utils import process_vision_info  # noqa: F401

        from ayase.runtime import shared_runtime_resource

        model_id = self.config.get("model_id", "chancharikm/qwen2.5-vl-7b-cam-motion")
        processor_id = self.config.get("processor_id", "Qwen/Qwen2.5-VL-7B-Instruct")

        def build():
            model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype="auto"
                )
                .to(self._device)
                .eval()
            )
            processor = AutoProcessor.from_pretrained(processor_id)
            return model, processor

        return shared_runtime_resource(
            self, ("qwen2.5-vl-camerabench", self._device), build
        )

    # -- processing ---------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            label, confidence, primitives = self._classify(sample)
            if label is not None and confidence is not None:
                _store_camera_motion_class(sample, label)
                if primitives:
                    sample.metadata["camera_motion_primitives"] = primitives
                sample.quality_metrics.camera_motion_class_confidence = round(
                    float(confidence), 6
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("camerabench processing failed: %s", exc)

        return sample

    def _classify(self, sample: Sample) -> Tuple[Optional[str], Optional[float], dict]:
        from PIL import Image

        from ayase.image import sample_frames

        frames = sample_frames(
            sample.path, max_frames=self.config.get("num_frames", 16), color="rgb"
        )
        if len(frames) < 2:
            return None, None, None
        # Read-only cache views -> contiguous copies before PIL/torch use.
        pil_frames = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]

        best_label: Optional[str] = None
        best_prob = -1.0
        primitives = {}
        for label, question in CAMERA_MOTION_LABELS.items():
            prob = self._score_yes(pil_frames, question)
            if prob is None:
                continue
            primitives[label] = round(prob, 6)
            if prob > best_prob:
                best_prob = prob
                best_label = label

        if best_label is None:
            return None, None, None
        return best_label, best_prob, primitives

    def _score_yes(self, pil_frames: List, question: str) -> Optional[float]:
        """Return P('Yes') for an official CameraBench primitive question.

        The prompt follows the benchmark's VQAScore template
        (``"{question} Please only answer Yes or No."``) and builds the video
        input via the canonical Qwen2.5-VL path
        (``qwen_vl_utils.process_vision_info``) so the frames and fps are
        packed exactly as the model expects, then reads the next-token
        distribution and sums the probability mass on the ``Yes`` token(s).
        """
        import torch
        from qwen_vl_utils import process_vision_info

        processor = self._processor
        prompt = f"{question} Please only answer Yes or No."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "fps": self.config.get("fps", 8.0),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            next_logits = outputs.logits[:, -1, :]
            probs = torch.softmax(next_logits.float(), dim=-1)
        yes_ids = self._yes_token_ids()
        if not yes_ids:
            return None
        return float(probs[0, yes_ids].sum().item())

    def _yes_token_ids(self) -> List[int]:
        tokenizer = self._processor.tokenizer
        ids = set()
        for word in ("Yes", " Yes", "yes", " yes"):
            enc = tokenizer.encode(word, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return list(ids)
