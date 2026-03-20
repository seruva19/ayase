"""AVQT — Apple Advanced Video Quality Tool.

Apple's perceptual video quality metric for content delivery.
Multi-scale perceptual comparison with human visual system modeling.

avqt_score — higher = better quality
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class AVQTModule(ReferenceBasedModule):
    name = "avqt"
    description = "Apple AVQT perceptual video quality (full-reference)"
    metric_field = "avqt_score"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._cli_available = False
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try AVQT CLI tool
        try:
            result = subprocess.run(
                ["avqt", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._cli_available = True
                self._backend = "cli"
                logger.info("AVQT (CLI) initialised")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("AVQT (heuristic) initialised — install avqt CLI for full model")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend == "cli":
            return self._compute_cli(sample_path, reference_path)
        return self._compute_heuristic(sample_path, reference_path)

    def _compute_cli(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Run AVQT CLI tool."""
        try:
            result = subprocess.run(
                ["avqt", "--ref", str(reference_path), "--dis", str(sample_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                # Parse score from output
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if "score" in line.lower() or "avqt" in line.lower():
                        parts = line.split()
                        for part in reversed(parts):
                            try:
                                return float(np.clip(float(part), 0.0, 1.0))
                            except ValueError:
                                continue
            return None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"AVQT CLI failed: {e}")
            return None

    def _read_frames(self, path: Path) -> list:
        """Read frames from video or image."""
        frames = []
        is_video = path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        }

        if is_video:
            cap = cv2.VideoCapture(str(path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)

        return frames

    def _compute_heuristic(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Heuristic: multi-scale perceptual comparison."""
        dist_frames = self._read_frames(sample_path)
        ref_frames = self._read_frames(reference_path)

        if not dist_frames or not ref_frames:
            return None

        n_frames = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_frames]
        ref_frames = ref_frames[:n_frames]

        frame_scores = []
        for i in range(n_frames):
            dist_gray = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Resize if needed
            if dist_gray.shape != ref_gray.shape:
                dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))

            # Multi-scale comparison (3 scales)
            scale_scores = []
            d_cur, r_cur = dist_gray.copy(), ref_gray.copy()

            for s in range(3):
                if d_cur.shape[0] < 8 or d_cur.shape[1] < 8:
                    break

                # SSIM-like comparison at this scale
                c1 = (0.01 * 255) ** 2
                c2 = (0.03 * 255) ** 2

                mu_d = cv2.GaussianBlur(d_cur, (11, 11), 1.5)
                mu_r = cv2.GaussianBlur(r_cur, (11, 11), 1.5)
                sigma_d2 = cv2.GaussianBlur(d_cur ** 2, (11, 11), 1.5) - mu_d ** 2
                sigma_r2 = cv2.GaussianBlur(r_cur ** 2, (11, 11), 1.5) - mu_r ** 2
                sigma_dr = cv2.GaussianBlur(d_cur * r_cur, (11, 11), 1.5) - mu_d * mu_r

                ssim_map = ((2 * mu_d * mu_r + c1) * (2 * sigma_dr + c2)) / (
                    (mu_d ** 2 + mu_r ** 2 + c1) * (sigma_d2 + sigma_r2 + c2)
                )
                scale_scores.append(float(np.mean(ssim_map)))

                # Downsample
                if d_cur.shape[0] > 16 and d_cur.shape[1] > 16:
                    d_cur = cv2.pyrDown(d_cur)
                    r_cur = cv2.pyrDown(r_cur)

            if scale_scores:
                # Weight finer scales more (perceptual importance)
                weights = [0.5, 0.3, 0.2][: len(scale_scores)]
                weights = np.array(weights) / sum(weights)
                frame_scores.append(float(np.sum(np.array(scale_scores) * weights)))

        if not frame_scores:
            return None

        score = float(np.mean(frame_scores))
        return float(np.clip(score, 0.0, 1.0))
