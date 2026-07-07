"""CameraBench camera-motion taxonomy classification (arXiv 2504.15376).

Classifies the dominant camera motion of a clip into the CameraBench
cinematographer taxonomy (static, pan left/right, tilt up/down, roll, zoom
in/out, dolly in/out, truck, pedestal, follow) using the fine-tuned Qwen2.5-VL
model released by the CameraBench authors:

  * checkpoint: ``chancharikm/qwen2.5-vl-7b-cam-motion``
    (32B / 72B variants exist: ``chancharikm/qwen2.5-vl-32b-cam-motion``,
    ``chancharikm/qwen2.5-vl-72b-cam-motion``)
  * processor:  ``Qwen/Qwen2.5-VL-7B-Instruct``
  * project:    https://linzhiqiu.github.io/papers/camerabench/
  * code:       https://github.com/sy77777en/CameraBench

Real backend only — there is no optical-flow heuristic classifier tier. If the
VLM cannot be loaded the module sets ``_backend = "unavailable"`` and leaves both
outputs unset.

On success the predicted label is stored in ``sample.metadata["camera_motion_class"]``
and the confidence in ``quality_metrics.camera_motion_class_confidence``.

Classification uses the model's generative yes/no scoring (per the model card):
for each taxonomy label it asks ``Does this video show "<description>"?`` and reads
the softmax probability of the ``Yes`` token from the next-token distribution; the
label with the highest probability is the prediction and that probability is the
reported confidence.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# CameraBench camera-motion taxonomy: {label: natural-language description used
# in the yes/no classification prompt}.
CAMERA_MOTION_LABELS = {
    "static": "the camera is static and not moving",
    "pan_left": "the camera pans left",
    "pan_right": "the camera pans right",
    "tilt_up": "the camera tilts upward",
    "tilt_down": "the camera tilts downward",
    "roll": "the camera rolls around its optical axis",
    "zoom_in": "the camera zooms in",
    "zoom_out": "the camera zooms out",
    "dolly_in": "the camera moves forward (dolly in)",
    "dolly_out": "the camera moves backward (dolly out)",
    "truck": "the camera trucks sideways (moves left or right)",
    "pedestal": "the camera pedestals vertically (moves up or down)",
    "follow": "the camera follows or tracks the subject",
}


def _store_camera_motion_class(sample: Sample, label: str) -> None:
    """Attach the predicted class to ``sample.metadata["camera_motion_class"]``.

    ``Sample`` is a pydantic model without a declared ``metadata`` field, so the
    dict is attached via ``object.__setattr__`` (bypassing pydantic validation).
    Readers can then use ``getattr(sample, "metadata", {})``; the value is
    process-local and not serialised by ``model_dump``.
    """
    try:
        meta = getattr(sample, "metadata", None)
        if not isinstance(meta, dict):
            meta = {}
            object.__setattr__(sample, "metadata", meta)
        meta["camera_motion_class"] = label
    except Exception:  # pragma: no cover - defensive
        logger.debug("camerabench: could not attach camera_motion_class metadata")


class CameraBenchModule(PipelineModule):
    name = "camerabench"
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
            label, confidence = self._classify(sample)
            if label is not None and confidence is not None:
                _store_camera_motion_class(sample, label)
                sample.quality_metrics.camera_motion_class_confidence = round(
                    float(confidence), 6
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("camerabench processing failed: %s", exc)

        return sample

    def _classify(self, sample: Sample) -> Tuple[Optional[str], Optional[float]]:
        from PIL import Image

        from ayase.image import sample_frames

        frames = sample_frames(
            sample.path, max_frames=self.config.get("num_frames", 16), color="rgb"
        )
        if len(frames) < 2:
            return None, None
        # Read-only cache views -> contiguous copies before PIL/torch use.
        pil_frames = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]

        best_label: Optional[str] = None
        best_prob = -1.0
        for label, description in CAMERA_MOTION_LABELS.items():
            prob = self._score_yes(pil_frames, description)
            if prob is not None and prob > best_prob:
                best_prob = prob
                best_label = label

        if best_label is None:
            return None, None
        return best_label, best_prob

    def _score_yes(self, pil_frames: List, description: str) -> Optional[float]:
        """Return P('Yes') for 'Does this video show "<description>"?'."""
        import torch

        processor = self._processor
        question = f'Does this video show "{description}"?'
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": self.config.get("fps", 8.0)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], videos=[pil_frames], return_tensors="pt").to(
            self._device
        )
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
