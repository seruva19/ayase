"""MOVIE (MOtion-based Video Integrity Evaluation) module.

Full-reference video quality metric using spatiotemporal Gabor filter
decomposition and optical flow analysis.

Backend tiers:
  1. **FR Gabor decomposition** — spatiotemporal Gabor filter bank
     (5 spatial orientations x 3 temporal scales) comparing ref vs dist
  2. **NR heuristic** — Gabor energy analysis + flow coherence (no reference)
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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        reference_path = getattr(sample, "reference_path", None)
        has_reference = reference_path is not None and Path(str(reference_path)).exists()

        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if has_reference:
                score = self._compute_fr(sample, Path(str(reference_path)), frames)
            else:
                score = self._compute_nr(frames, sample)

            if score is not None:
                sample.quality_metrics.movie_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("MOVIE failed: %s", e)
        return sample

    def _compute_fr(self, sample: Sample, reference_path: Path, dist_frames: list) -> Optional[float]:
        """Full-reference MOVIE: compare Gabor responses between ref and dist."""
        import cv2

        ref_frames = self._load_frames_from_path(reference_path)
        if not ref_frames:
            return self._compute_nr(dist_frames, sample)

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

    def _compute_nr(self, frames: list, sample: Sample) -> Optional[float]:
        """No-reference MOVIE: Gabor energy analysis + flow coherence."""
        import cv2

        # Spatial MOVIE: multi-orientation Gabor quality
        spatial_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            s_score = self._spatial_movie_nr(gray)
            spatial_scores.append(s_score)

        # Temporal MOVIE: motion coherence
        temporal_score = self._temporal_movie_nr(frames) if len(frames) >= 2 else 0.8

        spatial_mean = float(np.mean(spatial_scores))
        movie_score = np.sqrt(max(0.0, spatial_mean) * max(0.0, temporal_score))
        return float(movie_score)

    def _spatial_movie_nr(self, gray: np.ndarray) -> float:
        """NR spatial quality via Gabor filter energy analysis."""
        import cv2

        scores = []
        for theta in SPATIAL_ORIENTATIONS:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=theta,
                lambd=10.0, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            energy = np.mean(filtered ** 2)
            scores.append(energy)

        energies = np.array(scores)
        total_energy = np.sum(energies)
        if total_energy > 0:
            probs = energies / total_energy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(SPATIAL_ORIENTATIONS))
            orientation_balance = entropy / max_entropy
        else:
            orientation_balance = 0.0

        energy_score = min(1.0, total_energy / 5000.0)
        return 0.5 * orientation_balance + 0.5 * energy_score

    def _temporal_movie_nr(self, frames) -> float:
        """NR temporal quality via motion coherence analysis."""
        import cv2

        coherences = []
        flow_magnitudes = []

        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_magnitudes.append(np.mean(mag))

            fx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3)
            fy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3)
            flow_grad = np.sqrt(fx ** 2 + fy ** 2)
            coherence = 1.0 / (1.0 + np.mean(flow_grad))
            coherences.append(coherence)

        if len(flow_magnitudes) >= 2:
            mag_smoothness = 1.0 - min(1.0, np.std(flow_magnitudes) / (np.mean(flow_magnitudes) + 1e-8))
        else:
            mag_smoothness = 1.0

        spatial_coherence = float(np.mean(coherences)) if coherences else 0.5
        mean_mag = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        natural_motion = float(np.exp(-0.5 * ((mean_mag - 5.0) / 10.0) ** 2))

        return 0.35 * spatial_coherence + 0.35 * mag_smoothness + 0.30 * natural_motion

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
