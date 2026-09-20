"""Reference-free speech assessment with TorchAudio-SQUIM Objective.

This module estimates three metrics that normally require aligned clean speech:
STOI intelligibility (0-1, higher is better), wideband PESQ (approximately
1-4.6439, higher is better), and SI-SDR in dB (unbounded, higher is better).
The outputs are neural estimates, not intrusive metric computations and not
standards-compliant replacements for STOI, PESQ, or SI-SDR when a clean
reference is available.

The released model was trained on five-second, 16 kHz DNS 2020 speech signals.
Ayase decodes mono float32 audio at exactly 16 kHz without amplitude or loudness
normalization. Inputs longer than five seconds are covered by deterministic,
uniformly spaced five-second windows, including a tail-aligned final window;
at most ``max_windows`` are evaluated and their three predictions are averaged.
Shorter non-silent speech is evaluated at its native length. Missing audio,
near-silence, and audio shorter than ``min_duration_seconds`` are left unset.

Use this module for monaural speech, especially noisy or enhanced speech. Music,
sound effects, ambience, multichannel/spatial audio, silence, and subjective MOS
assessment are outside its intended domain. Domain-shifted speech may also be
less reliable than DNS-style speech-enhancement material.

Primary sources:
    - Kumar et al. (ICASSP 2023), "TorchAudio-Squim: Reference-less Speech
      Quality and Intelligibility measures in TorchAudio",
      https://arxiv.org/abs/2304.01448
    - https://pytorch.org/audio/stable/generated/torchaudio.pipelines.SQUIM_OBJECTIVE.html

Official checkpoint:
    - https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth
    - 29,584,237 bytes; SHA-256
      2c54586fea83fb5eb5394d710038ee89f55cab7011a5bf730bebed4c8777e828
    - Creative Commons Attribution 4.0 International (CC BY 4.0), as declared
      by TorchAudio from the DNS 2020 training-data license:
      https://github.com/microsoft/DNS-Challenge/blob/interspeech2020/master/LICENSE
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_WINDOW_SECONDS = 5.0
_WEIGHTS_URL = (
    "https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth"
)
_WEIGHTS_SHA256 = "2c54586fea83fb5eb5394d710038ee89f55cab7011a5bf730bebed4c8777e828"
_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class AudioSQUIMObjectiveModule(PipelineModule):
    """Estimate STOI, WB-PESQ, and SI-SDR from speech without a reference."""

    name = "audio_squim_objective"
    description = (
        "TorchAudio-SQUIM reference-free estimates of STOI, WB-PESQ, and SI-SDR"
    )
    default_config = {
        "device": "auto",
        "max_windows": 12,
        "min_duration_seconds": 1.0,
        "silence_rms_threshold": 1e-5,
    }
    models = [
        {
            "id": "squim_objective_dns2020.pth",
            "type": "local",
            "url": _WEIGHTS_URL,
            "task": "Reference-free estimation of STOI, WB-PESQ, and SI-SDR",
            "size": "29,584,237 bytes (28.213727 MiB)",
            "auto_download": True,
            "license": "CC BY 4.0",
            "notes": (
                "DNS 2020 weights; SHA-256 "
                + _WEIGHTS_SHA256
                + "; license https://github.com/microsoft/DNS-Challenge/blob/"
                "interspeech2020/master/LICENSE"
            ),
        },
    ]
    metric_info = {
        "squim_stoi_score": (
            "TorchAudio-SQUIM reference-free STOI estimate (0-1, higher=better)"
        ),
        "squim_pesq_score": (
            "TorchAudio-SQUIM reference-free WB-PESQ estimate "
            "(approximately 1-4.6439, higher=better)"
        ),
        "squim_si_sdr_score": (
            "TorchAudio-SQUIM reference-free SI-SDR estimate in dB "
            "(unbounded, higher=better)"
        ),
    }
    metric_groups = {
        "squim_stoi_score": "audio",
        "squim_pesq_score": "audio",
        "squim_si_sdr_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_windows = max(1, int(self.config.get("max_windows", 12)))
        self.min_duration_seconds = max(
            0.0, float(self.config.get("min_duration_seconds", 1.0))
        )
        self.silence_rms_threshold = max(
            0.0, float(self.config.get("silence_rms_threshold", 1e-5))
        )
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            from torchaudio.pipelines import SQUIM_OBJECTIVE

            from ayase.runtime import resolve_torch_device

            if int(SQUIM_OBJECTIVE.sample_rate) != _SAMPLE_RATE:
                raise RuntimeError(
                    "Unexpected SQUIM_OBJECTIVE sample rate: "
                    f"{SQUIM_OBJECTIVE.sample_rate}"
                )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = SQUIM_OBJECTIVE.get_model().to(self._device).eval()
            self._backend = "torchaudio:SQUIM_OBJECTIVE"
            logger.info("TorchAudio-SQUIM Objective initialized on %s", self._device)
        except Exception as e:
            self._model = None
            self._backend = "unavailable"
            logger.warning("TorchAudio-SQUIM Objective unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if self._model is None or sample.path.suffix.lower() in _IMAGE_SUFFIXES:
            return sample

        try:
            audio = load_audio(sample.path, target_sr=_SAMPLE_RATE, mono=True)
            windows = self._select_windows(audio)
            if not windows:
                return sample

            scores = [self._score_window(window) for window in windows]
            mean_scores = np.mean(np.asarray(scores, dtype=np.float64), axis=0)
            if not np.all(np.isfinite(mean_scores)):
                logger.warning("SQUIM returned non-finite scores for %s", sample.path)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.squim_stoi_score = round(float(mean_scores[0]), 4)
            sample.quality_metrics.squim_pesq_score = round(float(mean_scores[1]), 4)
            sample.quality_metrics.squim_si_sdr_score = round(float(mean_scores[2]), 4)
        except Exception as e:
            logger.warning("TorchAudio-SQUIM Objective failed for %s: %s", sample.path, e)

        return sample

    def _select_windows(self, audio: Optional[np.ndarray]) -> List[np.ndarray]:
        """Return deterministic speech windows without changing sample amplitudes."""
        if audio is None:
            return []

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        waveform = waveform.reshape(-1)

        min_samples = int(round(self.min_duration_seconds * _SAMPLE_RATE))
        if waveform.size < max(1, min_samples):
            return []

        window_samples = int(round(_WINDOW_SECONDS * _SAMPLE_RATE))
        if waveform.size <= window_samples:
            candidates = [waveform]
        else:
            available_windows = int(np.ceil(waveform.size / window_samples))
            count = min(self.max_windows, available_windows)
            last_start = waveform.size - window_samples
            if count == 1:
                starts = np.asarray([last_start], dtype=np.int64)
            else:
                starts = np.linspace(0, last_start, num=count, dtype=np.int64)
            candidates = [
                waveform[int(start) : int(start) + window_samples]
                for start in starts
            ]

        return [window for window in candidates if not self._is_near_silent(window)]

    def _is_near_silent(self, waveform: np.ndarray) -> bool:
        if waveform.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        return not np.isfinite(rms) or rms <= self.silence_rms_threshold

    def _score_window(self, waveform: np.ndarray) -> Tuple[float, float, float]:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float()
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.inference_mode():
            outputs = self._model(tensor)

        if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
            raise RuntimeError("SQUIM_OBJECTIVE must return [STOI, PESQ, SI-SDR]")

        values = tuple(float(output.detach().cpu().reshape(-1)[0].item()) for output in outputs)
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError("SQUIM_OBJECTIVE returned a non-finite score")
        return values
