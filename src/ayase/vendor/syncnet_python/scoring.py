"""Compute LSE-D and LSE-C for one tracked face.

Adapted from the MIT ``SyncNetInstance.py`` at the source commit recorded in
``SOURCE_MAP.md``. Operations use BGR 0-255 frames, default MFCC parameters,
five-frame image windows, and twenty-frame MFCC windows.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Tuple

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from .ffmpeg import run_ffmpeg

DEFAULT_BATCH_SIZE = 20
DEFAULT_VSHIFT = 15
FACE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
VIDEO_FRAME_RATE = 25
# One 25 fps frame spans 640 samples at 16 kHz and 4 MFCC frames (10 ms step).
SAMPLES_PER_VIDEO_FRAME = AUDIO_SAMPLE_RATE // VIDEO_FRAME_RATE
MFCC_FRAMES_PER_VIDEO_FRAME = 4
WINDOW_VIDEO_FRAMES = 5
WINDOW_MFCC_FRAMES = WINDOW_VIDEO_FRAMES * MFCC_FRAMES_PER_VIDEO_FRAME


class LipSyncEncoder(Protocol):
    def forward_lip(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward_aud(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class TrackScore:
    """LSE of one face track: distance (lower=better), confidence (higher=better), offset."""

    lse_d: float
    lse_c: float
    offset: int


def calc_pdist(feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 10) -> List[torch.Tensor]:
    """Distances between video and audio features for every shift within ``±vshift``."""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(0, len(feat1)):
        window = feat2p[i : i + win_size, :]
        dists.append(
            torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), window)
        )
    return dists


def lse_from_features(
    im_feat: torch.Tensor, cc_feat: torch.Tensor, vshift: int = DEFAULT_VSHIFT
) -> TrackScore:
    """LSE-D = min over shifts of the mean distance; LSE-C = median - min."""
    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    conf = torch.median(mdist) - minval
    return TrackScore(lse_d=float(minval), lse_c=float(conf), offset=vshift - int(minidx))


def extract_features(
    model: LipSyncEncoder,
    frames: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode sliding windows of ``frames`` ``(T, H, W, 3)`` BGR and int16 ``audio``.

    Returns per-window video and audio features on the CPU. Raises ``ValueError`` when
    the track is too short for a single window.
    """
    mfcc = np.stack(
        [np.array(coeffs) for coeffs in zip(*python_speech_features.mfcc(audio, sample_rate))]
    )
    cct = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0).astype(float))
    cct = cct.float()

    min_length = min(len(frames), math.floor(len(audio) / SAMPLES_PER_VIDEO_FRAME))
    lastframe = min_length - WINDOW_VIDEO_FRAMES
    if lastframe <= 0:
        raise ValueError(
            f"track too short: {min_length} frames, more than {WINDOW_VIDEO_FRAMES} are needed"
        )

    im_feat = []
    cc_feat = []
    with torch.no_grad():
        for i in range(0, lastframe, batch_size):
            window_starts = range(i, min(lastframe, i + batch_size))

            # (5, H, W, 3) -> (3, 5, H, W): channels-first video windows.
            im_batch = [
                torch.from_numpy(frames[v : v + WINDOW_VIDEO_FRAMES].astype(float))
                .float()
                .permute(3, 0, 1, 2)
                for v in window_starts
            ]
            im_out = model.forward_lip(torch.stack(im_batch, 0).to(device))
            im_feat.append(im_out.cpu())

            cc_batch = [
                cct[
                    :,
                    :,
                    :,
                    v * MFCC_FRAMES_PER_VIDEO_FRAME : v * MFCC_FRAMES_PER_VIDEO_FRAME
                    + WINDOW_MFCC_FRAMES,
                ]
                for v in window_starts
            ]
            cc_out = model.forward_aud(torch.cat(cc_batch, 0).to(device))
            cc_feat.append(cc_out.cpu())

    return torch.cat(im_feat, 0), torch.cat(cc_feat, 0)


def read_frame(path: Path) -> np.ndarray:
    """Read one BGR frame; raise on unreadable paths."""
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"OpenCV could not read frame {path}")
    return image


def load_track(crop_path: Path, work_dir: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split a face crop video into 224x224 frames and mono 16 kHz audio."""
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"
    run_ffmpeg(
        ["-y", "-i", str(crop_path), "-threads", "1", "-f", "image2", str(work_dir / "%06d.jpg")]
    )
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(crop_path),
            "-async",
            "1",
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )

    frame_files = sorted(work_dir.glob("*.jpg"))
    if not frame_files:
        raise RuntimeError(f"ffmpeg extracted no frames from {crop_path}")
    frames = np.stack([cv2.resize(read_frame(f), (FACE_SIZE, FACE_SIZE)) for f in frame_files])
    sample_rate, audio = wavfile.read(str(audio_path))
    return frames, audio, int(sample_rate)


def score_track(
    model: LipSyncEncoder,
    crop_path: Path,
    work_dir: Path,
    device: torch.device,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    vshift: int = DEFAULT_VSHIFT,
) -> TrackScore:
    """LSE-D and LSE-C of one face crop video."""
    frames, audio, sample_rate = load_track(crop_path, work_dir)
    im_feat, cc_feat = extract_features(
        model, frames, audio, sample_rate, batch_size=batch_size, device=device
    )
    return lse_from_features(im_feat, cc_feat, vshift=vshift)
