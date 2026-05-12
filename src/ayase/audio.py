"""Audio utility functions for Ayase."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .models import AudioMetadata

logger = logging.getLogger(__name__)


def get_audio_metadata(path: Path) -> Optional[AudioMetadata]:
    """Get metadata for the audio stream in a file.

    Args:
        path: Path to the media file

    Returns:
        AudioMetadata object or None if no audio or error
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if not data.get("streams"):
            return None

        stream = data["streams"][0]

        return AudioMetadata(
            sample_rate=int(stream.get("sample_rate", 0)),
            channels=int(stream.get("channels", 0)),
            bitrate=int(stream.get("bit_rate")) if stream.get("bit_rate") else None,
            codec=stream.get("codec_name", "unknown"),
            duration=float(stream.get("duration", 0.0)),
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as e:
        logger.debug(f"Could not get audio metadata for {path}: {e}")
        return None


def has_audio(path: Path) -> bool:
    """Check if a media file has an audio stream.

    Args:
        path: Path to the media file

    Returns:
        True if the file has audio, False otherwise
    """
    return get_audio_metadata(path) is not None


def load_audio(
    path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Load audio from an audio file or extract it from a video.

    The function returns float32 samples in roughly ``[-1, 1]``. It first tries
    ``soundfile``/``librosa`` directly, then falls back to ffmpeg extraction for
    containers where the audio stream cannot be decoded as a standalone file.
    """
    path = Path(path)

    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if mono and getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if duration is not None:
            audio = audio[: int(sr * duration)]
        if sr != target_sr:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except Exception:
                audio = _linear_resample(audio, sr, target_sr)
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        pass

    try:
        import librosa

        audio, _ = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            duration=duration,
        )
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        pass

    return extract_audio_with_ffmpeg(path, target_sr=target_sr, mono=mono, duration=duration)


def extract_audio_with_ffmpeg(
    path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Extract an audio stream with ffmpeg and return a mono float32 waveform."""
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = ["ffmpeg", "-y", "-i", str(path), "-vn"]
        if duration is not None:
            cmd.extend(["-t", str(float(duration))])
        cmd.extend([
            "-ac",
            "1" if mono else "2",
            "-ar",
            str(target_sr),
            "-sample_fmt",
            "s16",
            tmp.name,
        ])
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return None

        import soundfile as sf

        audio, _ = sf.read(tmp.name, dtype="float32")
        if mono and getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32)
    except (FileNotFoundError, ImportError) as e:
        logger.debug("ffmpeg audio extraction unavailable for %s: %s", path, e)
        return None
    except Exception as e:
        logger.debug("ffmpeg audio extraction failed for %s: %s", path, e)
        return None
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def audio_feature_summary(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Return compact signal features used by lightweight audio quality proxies."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return _empty_audio_summary()

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995))
    silence_ratio = float(np.mean(np.abs(audio) < 1e-4))

    frame = min(2048, max(256, int(sr * 0.05)))
    hop = max(1, frame // 2)
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
        frames = padded[None, :]
    else:
        frames = np.stack([
            audio[i:i + frame]
            for i in range(0, len(audio) - frame + 1, hop)
        ])
    windowed = frames * np.hanning(frame)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)

    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    dynamic_range = float(np.percentile(np.abs(audio), 95) - np.percentile(np.abs(audio), 5))
    noise_floor = float(np.percentile(frame_rms, 10))
    signal_level = float(np.percentile(frame_rms, 90))
    snr_proxy = float(20.0 * np.log10((signal_level + 1e-6) / (noise_floor + 1e-6)))

    mag_sum = spectrum.sum(axis=1) + 1e-8
    centroid = float(np.mean((spectrum * freqs[None, :]).sum(axis=1) / mag_sum))
    bandwidth = float(
        np.mean(np.sqrt(((freqs[None, :] - centroid) ** 2 * spectrum).sum(axis=1) / mag_sum))
    )
    flatness = float(np.mean(np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)))
    flux = 0.0
    if spectrum.shape[0] > 1:
        norm = spectrum / (np.linalg.norm(spectrum, axis=1, keepdims=True) + 1e-8)
        flux = float(np.mean(np.sqrt(np.sum(np.diff(norm, axis=0) ** 2, axis=1))))

    return {
        "peak": peak,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "dynamic_range": dynamic_range,
        "snr_proxy": snr_proxy,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
    }


def speech_quality_proxy(audio: np.ndarray, sr: int = 16000, mos: bool = False) -> float:
    """Map signal health indicators to a bounded speech-quality proxy."""
    f = audio_feature_summary(audio, sr=sr)
    snr = np.clip((f["snr_proxy"] + 5.0) / 35.0, 0.0, 1.0)
    level = np.clip(f["rms"] / 0.08, 0.0, 1.0) * np.clip(1.0 - max(f["rms"] - 0.35, 0.0) / 0.35, 0.0, 1.0)
    non_silence = 1.0 - np.clip(f["silence_ratio"], 0.0, 1.0)
    anti_noise = 1.0 - np.clip(f["spectral_flatness"], 0.0, 1.0)
    anti_clip = 1.0 - np.clip(f["clipping_ratio"] * 20.0, 0.0, 1.0)
    score = float(np.clip(
        0.35 * snr + 0.20 * level + 0.20 * non_silence + 0.15 * anti_noise + 0.10 * anti_clip,
        0.0,
        1.0,
    ))
    if mos:
        return float(1.0 + 4.0 * score)
    return score


def audio_distribution_features(audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """Return a stable feature vector for lightweight audio distribution metrics."""
    if audio is None or len(audio) < 256:
        return None
    f = audio_feature_summary(audio, sr=sr)
    return np.array([
        f["rms"],
        f["peak"],
        f["dynamic_range"],
        f["snr_proxy"],
        f["spectral_centroid"] / max(sr / 2, 1),
        f["spectral_bandwidth"] / max(sr / 2, 1),
        f["spectral_flatness"],
        f["spectral_flux"],
        f["clipping_ratio"],
        f["silence_ratio"],
    ], dtype=np.float64)


def _linear_resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr <= 0 or sr == target_sr or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    duration = len(audio) / float(sr)
    n_samples = max(1, int(duration * target_sr))
    source = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target = np.linspace(0.0, duration, num=n_samples, endpoint=False)
    return np.interp(target, source, audio).astype(np.float32)


def _empty_audio_summary() -> Dict[str, float]:
    return {
        "peak": 0.0,
        "rms": 0.0,
        "clipping_ratio": 0.0,
        "silence_ratio": 1.0,
        "dynamic_range": 0.0,
        "snr_proxy": 0.0,
        "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0,
        "spectral_flatness": 1.0,
        "spectral_flux": 0.0,
    }
