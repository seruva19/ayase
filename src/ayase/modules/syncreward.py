"""Audio-video synchronization quality using the published SyncReward model.

SyncReward evaluates an audiovisual video on its native 0-2 reward scale;
higher values indicate better synchronization. The canonical Stage-2 protocol
uses 18 overlapping 0.5-second segments at 0.25-second stride, centered within
the clip. Videos without sufficient decodable audio/video are left unscored.
"""

import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SyncRewardModule(PipelineModule):
    name = "syncreward"
    description = "SyncReward learned audio-video synchronization reward (0-2, higher is better)"
    provenance = "published"
    sources = {
        "syncreward_score": (
            "SyncReward, Guan et al. arXiv:2610.16338 — "
            "https://arxiv.org/abs/2610.16338"
        )
    }
    implementation_sources = (
        "../third_party/syncreward/data.py",
        "../third_party/syncreward/model.py",
    )
    default_config = {
        "checkpoint": None,
        "encoder_path": None,
        "models_dir": "models",
        "device": "auto",
        "num_segments": 18,
        "segment_duration": 0.5,
        "segment_stride": 0.25,
        "encode_batch_size": 18,
    }
    models = [
        {
            "id": "Guan123/SyncReward",
            "type": "huggingface",
            "revision": "04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a",
            "task": "SyncReward Stage-2 reward model",
            "auto_download": True,
        },
        {
            "id": "facebook/pe-av-large",
            "type": "huggingface",
            "revision": "0d24878d4107d64bef49e53602fc34ce6f94f6d8",
            "task": "PE-AV audio-video encoder",
            "auto_download": True,
        },
    ]
    metric_info = {
        "syncreward_score": "Native SyncReward score (0-2, higher=better)"
    }
    metric_groups = {"syncreward_score": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self.models_dir = self.config.get("models_dir", "models")
        self.num_segments = int(self.config.get("num_segments", 18))
        self.segment_duration = float(self.config.get("segment_duration", 0.5))
        self.segment_stride = float(self.config.get("segment_stride", 0.25))
        self.encode_batch_size = int(self.config.get("encode_batch_size", 18))
        self._backend = None
        self._model = None
        self._processor = None
        self._device = None
        self._torch = None
        self._load_av = None
        self._prepare_inputs = None
        self._processor_sample_rate = None
        self._min_video_frames = 2
        self._min_audio_samples = 4800
        self._dll_directory = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                conda_prefix = os.environ.get("CONDA_PREFIX")
                dll_dir = Path(conda_prefix) / "Library" / "bin" if conda_prefix else None
                if dll_dir is not None and dll_dir.is_dir():
                    self._dll_directory = os.add_dll_directory(str(dll_dir))

            import torch

            from ayase.config import download_hf_snapshot
            from ayase.third_party.syncreward.data import (
                MIN_AUDIO_SAMPLES,
                MIN_VIDEO_FRAMES,
                load_av,
            )
            from ayase.third_party.syncreward.model import (
                build_reward_model,
                load_processor,
                prepare_inputs,
                processor_sample_rate,
            )

            checkpoint = self.config.get("checkpoint")
            if checkpoint is None:
                snapshot = download_hf_snapshot(
                    "Guan123/SyncReward",
                    self.models_dir,
                    revision="04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a",
                    allow_patterns=["syncreward-stage2/*"],
                )
                checkpoint = Path(snapshot) / "syncreward-stage2"

            encoder_path = self.config.get("encoder_path")
            if encoder_path is None:
                encoder_path = download_hf_snapshot(
                    "facebook/pe-av-large",
                    self.models_dir,
                    revision="0d24878d4107d64bef49e53602fc34ce6f94f6d8",
                )

            requested_device = str(self.config.get("device", "auto"))
            device_name = (
                "cuda" if requested_device == "auto" and torch.cuda.is_available()
                else "cpu" if requested_device == "auto"
                else requested_device
            )
            device = torch.device(device_name)
            processor = load_processor(str(encoder_path))
            model = build_reward_model(
                str(encoder_path),
                checkpoint=str(checkpoint),
                num_segments=self.num_segments,
            )
            self._validate_model_num_segments(model)

            self._torch = torch
            self._device = device
            self._processor = processor
            self._model = model.to(device).eval()
            self._load_av = load_av
            self._prepare_inputs = prepare_inputs
            self._processor_sample_rate = processor_sample_rate
            self._min_video_frames = MIN_VIDEO_FRAMES
            self._min_audio_samples = MIN_AUDIO_SAMPLES
            self._backend = "syncreward"
        except Exception as exc:
            self._release_runtime()
            self._backend = "unavailable"
            logger.warning("SyncReward setup failed: %s", exc)

    def _validate_model_num_segments(self, model) -> None:
        if int(model.num_segments) != self.num_segments:
            raise ValueError(
                "SyncReward checkpoint num_segments "
                f"({model.num_segments}) does not match configured num_segments "
                f"({self.num_segments})"
            )

    def on_dispose(self) -> None:
        self._release_runtime()
        super().on_dispose()

    def _release_runtime(self) -> None:
        self._backend = None
        self._model = None
        self._processor = None
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._backend != "syncreward":
            return sample

        try:
            sample_rate = self._processor_sample_rate(self._processor)
            video, fps, audio = self._load_av(sample.path, sample_rate)
            if (
                video.shape[0] < self._min_video_frames
                or audio.shape[0] < self._min_audio_samples
            ):
                return sample

            args = SimpleNamespace(
                num_segments=self.num_segments,
                segment_duration=self.segment_duration,
                segment_stride=self.segment_stride,
            )
            inputs = self._prepare_inputs(
                self._processor,
                {"video": [video], "audio": [audio], "fps": [fps]},
                args,
                "center",
                self._device,
            )
            with self._torch.no_grad(), self._torch.autocast(
                self._device.type,
                dtype=self._torch.bfloat16,
                enabled=self._device.type == "cuda",
            ):
                output = self._model(inputs, encode_batch_size=self.encode_batch_size)
                score = float(output.float()[0])

            if not math.isfinite(score) or not 0.0 <= score <= 2.0:
                logger.warning("SyncReward returned invalid score: %r", score)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.syncreward_score = score
        except Exception as exc:
            logger.warning("SyncReward failed for %s: %s", sample.path, exc)
        return sample
