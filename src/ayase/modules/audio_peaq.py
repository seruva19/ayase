"""PEAQ (Perceptual Evaluation of Audio Quality) module.

ITU-R BS.1387 reference-based audio codec quality. Compares a degraded
signal against an original/reference, producing:
    * ODG (Objective Difference Grade) in [-4, 0] (0 = imperceptible)
    * DI  (Distortion Index, higher = better)

Tiered backend:
    1. ``peaqb-fast`` binary on PATH (real ITU-R BS.1387 implementation).
    2. Bark-band MSE psychoacoustic approximation.
    3. Log-spectral distance fallback.
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
        "backend": "auto",  # "auto" | "peaqb" | "bark" | "lsd"
        "mode": "basic",  # peaqb-fast: "basic" or "advanced"
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.target_sr = self.config.get("target_sr", 48000)
        self.backend_pref = self.config.get("backend", "auto")
        self.mode = self.config.get("mode", "basic")
        self._peaqb_path: Optional[str] = None
        self.active_backend = "lsd"

    def setup(self) -> None:
        if self.backend_pref in ("auto", "peaqb"):
            peaqb = shutil.which("peaqb") or shutil.which("peaqb-fast")
            if peaqb is not None:
                self._peaqb_path = peaqb
                self.active_backend = "peaqb"
                logger.info(f"PEAQ module using peaqb at {peaqb}")
                return
        if self.backend_pref in ("auto", "bark"):
            self.active_backend = "bark"
            logger.info("PEAQ module using Bark-band MSE approximation")
            return
        self.active_backend = "lsd"
        logger.info("PEAQ module using log-spectral distance fallback")

    def process(self, sample: Sample) -> Sample:
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
        ref_audio = ref_audio[:min_len]
        dist_audio = dist_audio[:min_len]

        try:
            if self.active_backend == "peaqb":
                odg, di = self._run_peaqb(reference, sample.path)
            elif self.active_backend == "bark":
                odg, di = self._bark_band_proxy(ref_audio, dist_audio)
            else:
                odg, di = self._lsd_proxy(ref_audio, dist_audio)
        except Exception as e:
            logger.debug(f"PEAQ scoring failed for {sample.path}: {e}")
            odg, di = self._lsd_proxy(ref_audio, dist_audio)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.peaq_odg = round(float(odg), 3)
        sample.quality_metrics.peaq_di = round(float(di), 3)
        return sample

    def _run_peaqb(self, ref_path: Path, dist_path: Path) -> Tuple[float, float]:
        # peaqb-fast emits ODG and DI on stdout. Some forks require WAV input;
        # we transcode via ffmpeg for safety.
        ref_wav = self._to_wav(ref_path)
        dist_wav = self._to_wav(dist_path)
        try:
            mode_arg = "-a" if self.mode == "advanced" else "-b"
            cmd = [self._peaqb_path, mode_arg, str(ref_wav), str(dist_wav)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            odg, di = _parse_peaqb_output(result.stdout + "\n" + result.stderr)
            return odg, di
        finally:
            for p in (ref_wav, dist_wav):
                if p is not None and p.exists():
                    p.unlink(missing_ok=True)

    def _bark_band_proxy(self, ref: np.ndarray, dist: np.ndarray) -> Tuple[float, float]:
        # Per-frame Bark-band energy difference, mapped to an ODG-like score.
        sr = self.target_sr
        frame_len = 2048
        hop = frame_len // 2
        n_frames = max(1, (len(ref) - frame_len) // hop)
        bark_edges = _bark_edges(sr, frame_len)
        diffs = []
        for i in range(n_frames):
            r = ref[i * hop: i * hop + frame_len]
            d = dist[i * hop: i * hop + frame_len]
            if len(r) < frame_len:
                break
            R = np.abs(np.fft.rfft(r * np.hanning(frame_len))) ** 2
            D = np.abs(np.fft.rfft(d * np.hanning(frame_len))) ** 2
            for lo, hi in bark_edges:
                if hi <= lo:
                    continue
                er = np.mean(R[lo:hi]) + 1e-12
                ed = np.mean(D[lo:hi]) + 1e-12
                diffs.append(abs(10 * np.log10(ed / er)))
        if not diffs:
            return 0.0, 1.0
        mean_db_diff = float(np.mean(diffs))
        # Map [0, 20] dB → [0, -4] ODG
        odg = float(np.clip(-mean_db_diff / 5.0, -4.0, 0.0))
        di = float(np.clip(2.0 - mean_db_diff / 5.0, -4.0, 4.0))
        return odg, di

    def _lsd_proxy(self, ref: np.ndarray, dist: np.ndarray) -> Tuple[float, float]:
        # Log-spectral distance — cheap, deterministic fallback.
        n_fft = min(len(ref), 4096)
        R = np.abs(np.fft.rfft(ref[:n_fft])) + 1e-9
        D = np.abs(np.fft.rfft(dist[:n_fft])) + 1e-9
        lsd = float(np.sqrt(np.mean((20 * np.log10(R) - 20 * np.log10(D)) ** 2)))
        # Map [0, 30] dB LSD → [0, -4] ODG
        odg = float(np.clip(-lsd / 7.5, -4.0, 0.0))
        di = float(np.clip(2.0 - lsd / 7.5, -4.0, 4.0))
        return odg, di

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


def _bark_edges(sr: int, n_fft: int):
    # Approximate critical bands; freq → bark = 13*atan(0.00076*f) + 3.5*atan((f/7500)**2)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    edges = []
    for b in range(int(barks.max())):
        lo = int(np.searchsorted(barks, b))
        hi = int(np.searchsorted(barks, b + 1))
        edges.append((lo, hi))
    return edges


def _parse_peaqb_output(text: str) -> Tuple[float, float]:
    odg = 0.0
    di = 1.0
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
    return odg, di
