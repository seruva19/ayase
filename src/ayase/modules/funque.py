"""FUNQUE (Fused Unified Quality Evaluator) module.

Full-reference quality metric that fuses SSIM, VIF, and DLM features.

Backend tiers:
  1. **funque package** — real FUNQUE from ``pip install funque``
     (``github.com/abhinaukumar/funque``)
  2. **Handcrafted FR** — OpenCV-based SSIM/VIF/DLM approximation (FR mode)
  3. **Handcrafted NR** — self-referencing proxy (NR mode, no reference needed)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FUNQUEModule(PipelineModule):
    name = "funque"
    description = "Fused quality evaluator (FUNQUE package, handcrafted FR, or NR fallback)"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._funque_available = False
        self._backend = "heuristic_nr"

    def setup(self) -> None:
        # Tier 1: Real FUNQUE package
        try:
            import funque as _funque
            self._funque_available = True
            self._backend = "funque"
            logger.info("FUNQUE loaded real funque package")
            return
        except (ImportError, Exception) as e:
            logger.info("funque package unavailable: %s", e)

        # Tiers 2/3 are always available (OpenCV-based)
        self._backend = "heuristic_fr"
        logger.info("FUNQUE using handcrafted approximation")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        reference_path = getattr(sample, "reference_path", None)
        has_reference = reference_path is not None and Path(str(reference_path)).exists()

        try:
            if self._funque_available and has_reference:
                self._process_funque(sample, Path(str(reference_path)))
            elif has_reference:
                self._process_fr_heuristic(sample, Path(str(reference_path)))
            else:
                self._process_nr_heuristic(sample)
        except Exception as e:
            logger.warning("FUNQUE failed: %s", e)
        return sample

    def _process_funque(self, sample: Sample, reference_path: Path) -> None:
        """Process using the real FUNQUE package."""
        import funque

        try:
            # FUNQUE API: compute quality score between reference and distorted
            result = funque.compute(
                str(reference_path),
                str(sample.path),
            )
            if isinstance(result, dict) and "funque" in result:
                score = float(result["funque"])
            elif isinstance(result, (int, float)):
                score = float(result)
            else:
                score = float(result)

            sample.quality_metrics.funque_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.info("FUNQUE package failed: %s, falling back to heuristic", e)
            self._process_fr_heuristic(sample, reference_path)

    def _process_fr_heuristic(self, sample: Sample, reference_path: Path) -> None:
        """Full-reference heuristic: compare sample against reference."""
        import cv2

        frames_dist = self._load_frames(sample)
        if not frames_dist:
            return

        # Load reference frames
        ref_sample_path = reference_path
        if ref_sample_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            frames_ref = self._load_video_frames(ref_sample_path)
        else:
            frame = cv2.imread(str(ref_sample_path))
            frames_ref = [frame] if frame is not None else []

        if not frames_ref:
            self._process_nr_heuristic(sample)
            return

        # Compare paired frames
        n_pairs = min(len(frames_dist), len(frames_ref))
        frame_scores = []
        for i in range(n_pairs):
            ref_gray = cv2.cvtColor(frames_ref[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            dist_gray = cv2.cvtColor(frames_dist[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Resize to match
            h, w = dist_gray.shape
            ref_gray = cv2.resize(ref_gray, (w, h))

            score = self._compute_fr_features(ref_gray, dist_gray)
            frame_scores.append(score)

        # Temporal pooling
        if len(frame_scores) > 1:
            sorted_scores = sorted(frame_scores)
            n = len(sorted_scores)
            p5 = sorted_scores[max(0, int(n * 0.05))]
            final = 0.4 * p5 + 0.4 * np.mean(frame_scores) + 0.2 * np.median(frame_scores)
        else:
            final = frame_scores[0]

        sample.quality_metrics.funque_score = float(np.clip(final, 0.0, 1.0))

    def _compute_fr_features(self, ref: np.ndarray, dist: np.ndarray) -> float:
        """Compute FR quality features between reference and distorted frames."""
        import cv2

        # 1. SSIM approximation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu_ref = cv2.GaussianBlur(ref, (11, 11), 1.5)
        mu_dist = cv2.GaussianBlur(dist, (11, 11), 1.5)
        sigma_ref_sq = cv2.GaussianBlur(ref ** 2, (11, 11), 1.5) - mu_ref ** 2
        sigma_dist_sq = cv2.GaussianBlur(dist ** 2, (11, 11), 1.5) - mu_dist ** 2
        sigma_ref_dist = cv2.GaussianBlur(ref * dist, (11, 11), 1.5) - mu_ref * mu_dist

        ssim_map = ((2 * mu_ref * mu_dist + C1) * (2 * sigma_ref_dist + C2)) / \
                   ((mu_ref ** 2 + mu_dist ** 2 + C1) * (sigma_ref_sq + sigma_dist_sq + C2))
        ssim_score = float(np.mean(ssim_map))

        # 2. VIF approximation (information fidelity)
        eps = 1e-10
        sigma_ref_sq = np.maximum(sigma_ref_sq, 0)
        g = sigma_ref_dist / (sigma_ref_sq + eps)
        sigma_v = sigma_dist_sq - g * sigma_ref_dist
        sigma_v = np.maximum(sigma_v, eps)
        sigma_n = 2.0  # Noise variance estimate
        vif_num = np.log2(1 + g ** 2 * sigma_ref_sq / (sigma_v + sigma_n))
        vif_den = np.log2(1 + sigma_ref_sq / sigma_n)
        vif_score = float(np.sum(vif_num) / (np.sum(vif_den) + eps))
        vif_score = min(1.0, max(0.0, vif_score))

        # 3. DLM (Detail Loss Measure) approximation
        hp_ref = ref - cv2.GaussianBlur(ref, (15, 15), 3.0)
        hp_dist = dist - cv2.GaussianBlur(dist, (15, 15), 3.0)
        detail_loss = np.mean(np.abs(hp_ref - hp_dist))
        dlm_score = max(0.0, 1.0 - detail_loss / 50.0)

        return 0.35 * ssim_score + 0.35 * vif_score + 0.30 * dlm_score

    def _process_nr_heuristic(self, sample: Sample) -> None:
        """No-reference heuristic (self-referencing)."""
        import cv2

        frames = self._load_frames(sample)
        if not frames:
            return

        frame_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            score = self._compute_nr_features(gray)
            frame_scores.append(score)

        if len(frame_scores) > 1:
            sorted_scores = sorted(frame_scores)
            n = len(sorted_scores)
            p5 = sorted_scores[max(0, int(n * 0.05))]
            final = 0.4 * p5 + 0.4 * np.mean(frame_scores) + 0.2 * np.median(frame_scores)
        else:
            final = frame_scores[0]

        sample.quality_metrics.funque_score = float(np.clip(final, 0.0, 1.0))

    def _compute_nr_features(self, gray: np.ndarray) -> float:
        """Compute NR quality features from a grayscale frame."""
        import cv2

        h, w = gray.shape

        # Multi-scale SSIM-like luminance/contrast
        ssim_features = []
        current = gray.copy()
        for scale in range(3):
            mu = cv2.GaussianBlur(current, (11, 11), 1.5)
            mu_sq = mu ** 2
            sigma_sq = cv2.GaussianBlur(current ** 2, (11, 11), 1.5) - mu_sq
            sigma_sq = np.maximum(sigma_sq, 0)
            lum = np.mean(mu) / 255.0
            con = np.mean(np.sqrt(sigma_sq + 1e-8)) / 128.0
            ssim_features.extend([lum, min(1.0, con)])
            current = cv2.resize(current, (max(1, w // (2 ** (scale + 1))),
                                           max(1, h // (2 ** (scale + 1)))))

        # VIF-like information content
        vif_features = []
        current = gray.copy()
        for scale in range(3):
            mu = cv2.GaussianBlur(current, (5, 5), 1.0)
            var = cv2.GaussianBlur((current - mu) ** 2, (5, 5), 1.0)
            info = np.log2(1.0 + np.mean(var))
            vif_features.append(min(1.0, info / 12.0))
            current = cv2.resize(current, (max(1, current.shape[1] // 2),
                                           max(1, current.shape[0] // 2)))

        # DLM-like detail energy
        hp = gray - cv2.GaussianBlur(gray, (15, 15), 3.0)
        detail_energy = np.mean(hp ** 2)
        detail_score = min(1.0, detail_energy / 500.0)

        # Gradient structural features
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_coherence = np.std(grad_mag) / (np.mean(grad_mag) + 1e-8)
        structural = min(1.0, 1.0 / (1.0 + abs(grad_coherence - 1.0)))

        ssim_score = np.mean(ssim_features)
        vif_score = np.mean(vif_features)

        return float(0.30 * ssim_score + 0.25 * vif_score + 0.25 * detail_score + 0.20 * structural)

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

    def _load_video_frames(self, path: Path) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
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
