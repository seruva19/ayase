"""DNSMOS (Deep Noise Suppression MOS) module.

DNSMOS is Microsoft's non-intrusive speech quality metric that predicts
MOS scores without needing a reference signal. It outputs three scores:
  SIG  — signal quality (1-5)
  BAK  — background noise quality (1-5)
  OVRL — overall quality (1-5)

Based on ITU-T P.835 framework.

Requires ``torchmetrics[audio]`` or the Microsoft DNS-Challenge package.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DNSMOSModule(PipelineModule):
    name = "dnsmos"
    description = "DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try torchmetrics DNSMOS
        try:
            import torchmetrics.audio as tm_audio

            self._metric_cls = getattr(
                tm_audio,
                "DeepNoiseSuppressionMeanOpinionScore",
                None,
            )
            if self._metric_cls is None:
                self._metric_cls = getattr(tm_audio, "DeepNoiseSuppression", None)
            if self._metric_cls is None:
                raise ImportError("torchmetrics.audio DNSMOS class not found")
            self._backend = "torchmetrics"
            self._ml_available = True
            logger.info("DNSMOS module initialised (torchmetrics)")
            return
        except Exception as e:
            logger.debug("DNSMOS torchmetrics setup failed: %s", e)

        logger.warning("DNSMOS requires torchmetrics[audio]")

    def _extract_audio(self, sample_path: Path) -> Optional[tuple]:
        """Extract audio waveform + sample rate.

        Fast path: pipe PCM directly out of ffmpeg (no tempfile, no librosa
        audioread fallback). Falls back to soundfile / librosa for pure audio
        formats or environments without ffmpeg.
        """
        ffmpeg_result = self._ffmpeg_pipe_pcm(sample_path, sr=16000)
        if ffmpeg_result is not None:
            return ffmpeg_result

        try:
            import soundfile as sf
            audio, sr = sf.read(str(sample_path))
            return audio, sr
        except Exception:
            pass

        try:
            import librosa
            audio, sr = librosa.load(str(sample_path), sr=16000, mono=True)
            return audio, sr
        except Exception:
            pass

        return None

    def _ffmpeg_pipe_pcm(self, path: Path, sr: int = 16000) -> Optional[tuple]:
        """Decode mono float32 PCM from any container via ffmpeg stdout pipe."""
        import subprocess
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-nostdin",
                    "-i", str(path),
                    "-vn", "-ac", "1", "-ar", str(sr),
                    "-f", "s16le", "-",
                ],
                capture_output=True, timeout=60,
            )
            if proc.returncode != 0 or not proc.stdout:
                return None
            import numpy as np
            audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size == 0:
                return None
            return audio, sr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception as e:
            logger.debug(f"ffmpeg audio extraction failed for {path}: {e}")
            return None

    def _compute_torchmetrics(self, audio, sr: int) -> Optional[dict]:
        try:
            import torch

            metric = self._metric_cls(fs=sr, personalized=False)
            if not isinstance(audio, torch.Tensor):
                import numpy as np
                if isinstance(audio, np.ndarray):
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio = torch.from_numpy(audio).float().unsqueeze(0)
                else:
                    return None

            with torch.no_grad():
                result = metric(audio)

            if isinstance(result, dict):
                return {
                    "sig": float(result.get("SIG", result.get("sig", 0))),
                    "bak": float(result.get("BAK", result.get("bak", 0))),
                    "ovrl": float(result.get("OVRL", result.get("ovrl", 0))),
                }
            if hasattr(result, "detach"):
                values = result.detach().cpu().reshape(-1).tolist()
            elif isinstance(result, (list, tuple)):
                values = list(result)
            else:
                values = [result]
            if len(values) >= 3:
                return {
                    "sig": float(values[0]),
                    "bak": float(values[1]),
                    "ovrl": float(values[2]),
                }
            if len(values) == 1:
                value = float(values[0])
                return {"sig": value, "bak": value, "ovrl": value}
            return None
        except Exception as e:
            logger.debug(f"DNSMOS torchmetrics failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        result = self._extract_audio(sample.path)
        if result is None:
            return sample

        audio, sr = result

        try:
            if self._backend == "torchmetrics":
                scores = self._compute_torchmetrics(audio, sr)
            else:
                scores = None

            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dnsmos_overall = scores.get("ovrl")
            sample.quality_metrics.dnsmos_sig = scores.get("sig")
            sample.quality_metrics.dnsmos_bak = scores.get("bak")
            logger.debug(
                f"DNSMOS for {sample.path.name}: "
                f"SIG={scores.get('sig', 0.0):.2f} BAK={scores.get('bak', 0.0):.2f} OVRL={scores.get('ovrl', 0.0):.2f}"
            )
        except Exception as e:
            logger.error(f"DNSMOS failed: {e}")
        return sample
