"""VMBench Motion Smoothness (MSS) helpers — vendored verbatim.

Source: github.com/AMAP-ML/VMBench, ``motion_smoothness_score.py`` (Apache-2.0).
MSS scores every frame's local quality with Q-Align over a centred sliding
window, flags frames where the score jumps sharply (a motion artifact), and
reports the fraction of clean frames. The Q-Align backend and the per-window
scoring live in the ayase module; only the pure-NumPy artifact detection, the
PAS-derived threshold, and the sliding-window index builder are kept here.

    MSS = 1 - len(artifact_frames) / len(scores)          (0-1, higher=better)
"""

from typing import List

import numpy as np
from PIL import Image


def set_threshold(camera_movement):
    """PAS (perceptible amplitude) -> score-jump threshold. None -> 0.01."""
    if camera_movement is None:
        return 0.01
    if camera_movement < 0.1:
        return 0.01
    elif 0.1 <= camera_movement < 0.3:
        return 0.015
    elif 0.3 <= camera_movement < 0.5:
        return 0.025
    else:
        return 0.03


def get_artifacts_frames(scores, threshold=0.025):
    """Frames whose adjacent per-window quality jumps by more than threshold
    (both sides of the jump are counted)."""
    score_diffs = np.abs(np.diff(scores))
    artifact_indices = np.where(score_diffs > threshold)[0]
    artifacts_frames = np.unique(np.concatenate([artifact_indices, artifact_indices + 1]))
    return artifacts_frames


def sliding_window_groups(frames_rgb: List[np.ndarray], window_size: int = 5) -> List[List[Image.Image]]:
    """Build one centred ``window_size``-frame window (as PIL images) per frame,
    edge-padded exactly as VMBench's ``load_video_sliding_window``."""
    total_frames = len(frames_rgb)
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend
    groups: List[List[Image.Image]] = []
    for current_frame in range(total_frames):
        start_frame = max(0, current_frame - left_extend)
        end_frame = min(total_frames, current_frame + right_extend + 1)
        frame_indices = list(range(start_frame, end_frame))
        while len(frame_indices) < window_size:
            if start_frame == 0:
                frame_indices.append(frame_indices[-1])
            else:
                frame_indices.insert(0, frame_indices[0])
        if current_frame < left_extend:
            groups.append([Image.fromarray(frames_rgb[frame_indices[0]])] * window_size)
        else:
            groups.append([Image.fromarray(frames_rgb[i]) for i in frame_indices])
    return groups
