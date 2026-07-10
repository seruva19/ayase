"""VMBench Motion Smoothness Score (MSS) — Q-Align frame-quality jump detection.

Faithful port of VMBench's Motion Smoothness Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076). MSS scores every frame's local quality with Q-Align over a
centred 5-frame sliding window, then flags frames where that quality jumps
sharply between neighbours — the signature of motion jitter/discontinuity —
and reports the fraction of clean frames:

    MSS = 1 - len(artifact_frames) / len(scores)          (0-1, higher=smoother)

The Q-Align backend is the in-tree vendored mPLUG-Owl2 (``ayase.vendor.q_align``,
transformers-4.56 compatible, weights from q-future/one-align on HF). VMBench's
MSS scorer uses the level weights [1, 0.75, 0.5, 0.25, 0] (0-1 per window), and
the artifact threshold is derived from the clip's Perceptible Amplitude Score
(``perceptible_amplitude_score``) when available, else the default 0.01.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# VMBench MSS level weights (excellent..bad) — score per window in [0, 1].
_MSS_WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]


class VMBenchMotionSmoothnessModule(PipelineModule):
    name = "vmbench_mss"
    description = "VMBench Motion Smoothness — Q-Align per-frame quality-jump detection (0-1, higher=smoother)"
    default_config = {
        "model_name": "q-future/one-align",
        "dtype": "float16",
        "device": "auto",
        "window_size": 5,
        "max_frames": 64,       # consecutive frames scored (VMBench clips ~49f)
        "batch_windows": 8,     # windows scored per Q-Align forward (VRAM cap)
        "warn_threshold": 0.6,
    }
    metric_info = {
        "vmbench_mss": "VMBench Motion Smoothness (Q-Align quality-jump; 0-1, higher=smoother)",
    }
    metric_groups = {
        "vmbench_mss": "motion",
    }
    models = [
        {"id": "q-future/one-align", "type": "huggingface",
         "url": "https://huggingface.co/q-future/one-align",
         "task": "OneAlign (mPLUG-Owl2) per-frame quality scorer",
         "notes": "Loaded via vendored ayase.vendor.q_align"},
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._device = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                     "float32": torch.float32}.get(self.config.get("dtype", "float16"), torch.float16)

            # In-tree vendored Q-Align (registers mplug_owl2, no trust_remote_code).
            import ayase.vendor.q_align  # noqa: F401
            from ayase.vendor.q_align.modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM

            self._model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                self.config.get("model_name", "q-future/one-align"),
                torch_dtype=dtype,
                trust_remote_code=False,
                device_map="auto" if self._device.type == "cuda" else None,
            )
            if self._device.type != "cuda":
                self._model = self._model.to(self._device)
            self._model.eval()
            # VMBench MSS scores with 0-1 level weights (not the default 1-5).
            self._model.weight_tensor = torch.tensor(
                _MSS_WEIGHTS, dtype=dtype, device=self._model.device
            )
            self._ml_available = True
            logger.info("VMBench MSS initialised (vendored Q-Align backend)")
        except ImportError as e:
            logger.warning("VMBench MSS unavailable (torch/transformers missing): %s", e)
        except Exception as e:
            logger.warning("Failed to setup VMBench MSS: %s", e)

    def _read_frames(self, path: str, max_frames: int) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(path)
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def _score_windows(self, windows) -> List[float]:
        """Q-Align quality score (0-1) for each sliding window."""
        import torch

        batch = int(self.config.get("batch_windows", 8))
        scores: List[float] = []
        for i in range(0, len(windows), batch):
            chunk = windows[i:i + batch]
            with torch.no_grad():
                out = self._model.score(chunk, task_="quality", input_="video")
            scores.extend([float(x) for x in out.reshape(-1).tolist()])
        return scores

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        frames = self._read_frames(str(sample.path), self.config.get("max_frames", 64))
        if len(frames) < 3:
            return sample

        from ayase.vendor.vmbench.motion_smoothness_utils import (
            sliding_window_groups,
            get_artifacts_frames,
            set_threshold,
        )

        windows = sliding_window_groups(frames, int(self.config.get("window_size", 5)))
        try:
            scores = self._score_windows(windows)
        except Exception as e:
            logger.warning("VMBench MSS scoring failed for %s: %s", sample.path, e)
            return sample
        if len(scores) < 2:
            return sample

        # Threshold from the clip's Perceptible Amplitude Score if it was computed
        # earlier in the pipeline; otherwise VMBench's default (0.01).
        pas = None
        if sample.quality_metrics is not None:
            pas = getattr(sample.quality_metrics, "perceptible_amplitude_score", None)
        threshold = set_threshold(pas)

        artifacts = get_artifacts_frames(np.asarray(scores), threshold)
        mss = float(1.0 - len(artifacts) / len(scores))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.vmbench_mss = mss
        logger.debug("VMBench MSS for %s: %.3f (%d/%d artifact frames, thr=%.3f)",
                     sample.path.name, mss, len(artifacts), len(scores), threshold)
        return sample
