"""PEAQ (Perceptual Evaluation of Audio Quality) module.

ITU-R BS.1387 reference-based audio codec quality. Compares a degraded
signal against an original/reference, producing:
    * ODG (Objective Difference Grade) in [-4, 0] (0 = imperceptible)
    * DI  (Distortion Index, higher = better)

Backend: the real ``peaqb`` / ``peaqb-fast`` binary on PATH (an ITU-R BS.1387
implementation). When the binary is not available the PEAQ metrics are left
unset — there is no psychoacoustic approximation stand-in.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioPEAQModule(PipelineModule):
    name = "audio_peaq"
    description = "PEAQ reference-based audio codec quality (ITU-R BS.1387)"
    default_config = {
        "target_sr": 48000,  # PEAQ is defined for 48 kHz
        "mode": "basic",  # peaqb-fast: "basic" or "advanced"
    }
    metric_groups = {
        "peaq_di": "audio",
        "peaq_odg": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 48000)
        self.mode = self.config.get("mode", "basic")
        self._peaqb_path: Optional[str] = None
        self.active_backend = "unavailable"
        self._backend = "unavailable"

    def setup(self) -> None:
        peaqb = shutil.which("peaqb") or shutil.which("peaqb-fast")
        if peaqb is not None:
            self._peaqb_path = peaqb
            self.active_backend = "peaqb"
            self._backend = "binary:peaqb"
            logger.info(f"PEAQ module using peaqb at {peaqb}")
            return
        logger.warning(
            "PEAQ unavailable: no 'peaqb'/'peaqb-fast' binary on PATH; "
            "peaq_odg/peaq_di will be left unset."
        )

    def process(self, sample: Sample) -> Sample:
        if self.active_backend != "peaqb":
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        ref_audio = self._load_audio(reference)
        dist_audio = self._load_audio(sample.path)
        if ref_audio is None or dist_audio is None:
            return sample

        min_len = min(len(ref_audio), len(dist_audio))
        if min_len < self.target_sr // 4:  # need at least 0.25s
            return sample

        try:
            result = self._run_peaqb(reference, sample.path)
        except Exception as e:
            logger.warning(f"PEAQ scoring failed for {sample.path}: {e}")
            return sample
        if result is None:
            return sample
        odg, di = result

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.peaq_odg = round(float(odg), 3)
        sample.quality_metrics.peaq_di = round(float(di), 3)
        return sample

    def _run_peaqb(self, ref_path: Path, dist_path: Path) -> Optional[Tuple[float, float]]:
        # peaqb-fast emits ODG and DI on stdout. Some forks require WAV input;
        # we transcode via ffmpeg for safety.
        ref_wav = self._to_wav(ref_path)
        dist_wav = self._to_wav(dist_path)
        if ref_wav is None or dist_wav is None:
            for p in (ref_wav, dist_wav):
                if p is not None and p.exists():
                    p.unlink(missing_ok=True)
            return None
        try:
            mode_arg = "-a" if self.mode == "advanced" else "-b"
            cmd = [self._peaqb_path, mode_arg, str(ref_wav), str(dist_wav)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return _parse_peaqb_output(result.stdout + "\n" + result.stderr)
        finally:
            for p in (ref_wav, dist_wav):
                if p is not None and p.exists():
                    p.unlink(missing_ok=True)

    def _to_wav(self, path: Path) -> Optional[Path]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(self.target_sr),
                "-sample_fmt", "s16", tmp.name,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            return Path(tmp.name)
        except FileNotFoundError:
            return None

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            if sr != self.target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)
        except ImportError:
            return None
        except Exception:
            return None


def _parse_peaqb_output(text: str) -> Optional[Tuple[float, float]]:
    odg: Optional[float] = None
    di: Optional[float] = None
    for line in text.splitlines():
        low = line.lower()
        if "odg" in low:
            for tok in line.replace("=", " ").split():
                try:
                    odg = float(tok)
                    break
                except ValueError:
                    continue
        elif ("distortion index" in low) or low.startswith("di"):
            for tok in line.replace("=", " ").split():
                try:
                    di = float(tok)
                    break
                except ValueError:
                    continue
    if odg is None and di is None:
        return None
    return (odg if odg is not None else 0.0, di if di is not None else 0.0)
