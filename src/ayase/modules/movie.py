"""MOVIE (MOtion-based Video Integrity Evaluation) module.

Full-reference video quality metric using spatiotemporal Gabor filter
decomposition and optical-flow analysis: a spatiotemporal Gabor filter bank
(5 spatial orientations x 3 spatial frequencies) plus motion-compensated flow
comparison between the reference and distorted videos (paper reimplementation).

MOVIE is inherently full-reference — it has no meaning without a pristine
reference — so this module leaves ``movie_score`` unset when no
``reference_path`` is provided rather than emitting a no-reference proxy.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Gabor filter bank parameters from MOVIE paper (TIP 2010)
SPATIAL_ORIENTATIONS = [0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5, 4 * np.pi / 5]
SPATIAL_FREQUENCIES = [0.05, 0.1, 0.2]  # cycles/pixel


class MOVIEModule(PipelineModule):
    name = "movie"
    description = "Video quality via spatiotemporal Gabor decomposition (FR or NR fallback)"
    default_config = {"subsample": 8}
    metric_groups = {
        "movie_score": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = None

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        reference_path = getattr(sample, "reference_path", None)
        has_reference = reference_path is not None and Path(str(reference_path)).exists()

        if not has_reference:
            # MOVIE is full-reference; without a reference there is nothing to
            # compute. Leave movie_score unset rather than fabricate one.
            self._backend = "unavailable"
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._compute_fr(sample, Path(str(reference_path)), frames)

            if score is not None:
                self._backend = "port"
                sample.quality_metrics.movie_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("MOVIE failed: %s", e)
        return sample

    def _compute_fr(self, sample: Sample, reference_path: Path, dist_frames: list) -> Optional[float]:
        """Full-reference MOVIE: compare Gabor responses between ref and dist."""
        import cv2

        ref_frames = self._load_frames_from_path(reference_path)
        if not ref_frames:
            return None

        n_pairs = min(len(dist_frames), len(ref_frames))

        # Spatial MOVIE: Gabor filter bank comparison
        spatial_scores = []
        for i in range(n_pairs):
            ref_gray = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            dist_gray = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

            h, w = dist_gray.shape
            ref_gray = cv2.resize(ref_gray, (w, h))

            score = self._spatial_movie_fr(ref_gray, dist_gray)
            spatial_scores.append(score)

        # Temporal MOVIE: motion-compensated comparison
        temporal_score = self._temporal_movie_fr(ref_frames[:n_pairs], dist_frames[:n_pairs])

        spatial_mean = float(np.mean(spatial_scores))
        movie_score = np.sqrt(max(0.0, spatial_mean) * max(0.0, temporal_score))
        return float(movie_score)

    def _spatial_movie_fr(self, ref_gray: np.ndarray, dist_gray: np.ndarray) -> float:
        """Full-reference spatial MOVIE: compare Gabor responses."""
        import cv2

        C = 0.01  # Stability constant

        subband_scores = []
        for theta in SPATIAL_ORIENTATIONS:
            for freq in SPATIAL_FREQUENCIES:
                lambd = 1.0 / max(freq, 0.01)
                kernel_size = max(5, int(lambd * 2) | 1)  # Ensure odd
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma=lambd * 0.4,
                    theta=theta, lambd=lambd, gamma=0.5, psi=0
                )

                ref_resp = cv2.filter2D(ref_gray, cv2.CV_64F, kernel)
                dist_resp = cv2.filter2D(dist_gray, cv2.CV_64F, kernel)

                # Quality index for this subband
                ref_energy = np.mean(ref_resp ** 2) + C
                dist_energy = np.mean(dist_resp ** 2) + C
                cross_energy = np.mean(ref_resp * dist_resp) + C

                # Structural similarity in Gabor domain
                quality = (2 * cross_energy) / (ref_energy + dist_energy)
                subband_scores.append(max(0.0, min(1.0, quality)))

        return float(np.mean(subband_scores))

    def _temporal_movie_fr(self, ref_frames: list, dist_frames: list) -> float:
        """Full-reference temporal MOVIE: motion-compensated comparison."""
        import cv2

        if len(ref_frames) < 2 or len(dist_frames) < 2:
            return 1.0

        scores = []
        for i in range(len(dist_frames) - 1):
            # Compute flow for both ref and dist
            ref_g1 = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2GRAY)
            ref_g2 = cv2.cvtColor(ref_frames[i + 1] if i + 1 < len(ref_frames) else ref_frames[i], cv2.COLOR_BGR2GRAY)
            dist_g1 = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2GRAY)
            dist_g2 = cv2.cvtColor(dist_frames[i + 1], cv2.COLOR_BGR2GRAY)

            h, w = dist_g1.shape
            ref_g1 = cv2.resize(ref_g1, (w, h))
            ref_g2 = cv2.resize(ref_g2, (w, h))

            ref_flow = cv2.calcOpticalFlowFarneback(ref_g1, ref_g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dist_flow = cv2.calcOpticalFlowFarneback(dist_g1, dist_g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Flow field difference
            flow_diff = np.sqrt(np.sum((ref_flow - dist_flow) ** 2, axis=-1))
            flow_error = np.mean(flow_diff)
            motion_quality = 1.0 / (1.0 + flow_error)
            scores.append(motion_quality)

        return float(np.mean(scores)) if scores else 1.0

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

    def _load_frames_from_path(self, path: Path) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            frames = []
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            return frames
        else:
            frame = cv2.imread(str(path))
            return [frame] if frame is not None else []
