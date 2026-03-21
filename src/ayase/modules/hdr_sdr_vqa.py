"""HDR/SDR Video Quality Assessment module.

Dynamic range-aware quality assessment that adjusts metrics based on
whether content is HDR or SDR. Uses tone-mapping aware quality metrics for HDR.
"""

import logging

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HDRSDRVQAModule(PipelineModule):
    name = "hdr_sdr_vqa"
    description = "HDR/SDR-aware video quality assessment"
    default_config = {
        # OpenCV-based
        "subsample": 5,  # Process every Nth frame
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)

    def setup(self) -> None:
        pass

    def _detect_hdr(self, frame: np.ndarray, video_path=None) -> bool:
        """Detect if content is HDR based on pixel dtype and (for video) ffprobe metadata."""
        # Fast path: dtype-based detection
        # Note: cv2.VideoCapture usually returns uint8, but some builds
        # may return float32 for HDR content.
        if frame.dtype in (np.uint16, np.float32, np.float64):
            return True

        # For video files, probe color space metadata via ffprobe
        if video_path is not None:
            try:
                import subprocess
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=color_space,color_transfer,color_primaries",
                        "-of", "csv=p=0",
                        str(video_path),
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                probe_out = result.stdout.lower()
                hdr_indicators = ("bt2020", "smpte2084", "arib-std-b67", "bt2020nc")
                if any(ind in probe_out for ind in hdr_indicators):
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                # ffprobe not available — rely on dtype check only
                pass

        return False

    def _compute_hdr_quality(self, frames: list) -> float:
        """Compute HDR-specific quality metrics."""
        # HDR quality: check for proper tone mapping, no clipping
        quality_scores = []

        for frame in frames:
            # Convert to luminance
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = frame.astype(np.float32)

            # Normalize if uint16
            if frame.dtype == np.uint16:
                gray = gray / 65535.0
            else:
                gray = gray / 255.0

            # Check clipping (bad for HDR)
            clipping = ((gray < 0.01).sum() + (gray > 0.99).sum()) / gray.size

            # Check dynamic range utilization
            hist, _ = np.histogram(gray, bins=256, range=(0, 1))
            hist_norm = hist / hist.sum()
            # Entropy of histogram (higher = better use of dynamic range)
            hist_norm = hist_norm[hist_norm > 0]
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            entropy_norm = entropy / 8.0  # Normalize (max entropy = 8 for 256 bins)

            # Quality = high entropy, low clipping
            quality = (1.0 - clipping) * entropy_norm
            quality_scores.append(quality)

        return float(np.mean(quality_scores)) if quality_scores else 0.5

    def _compute_sdr_quality(self, frames: list) -> float:
        """Compute SDR-specific quality metrics."""
        quality_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # SDR quality: brightness, contrast, sharpness
            brightness = gray.mean() / 255.0
            contrast = gray.std() / 128.0

            # Sharpness (Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = min(laplacian.var() / 1000.0, 1.0)

            # Combined score
            quality = (brightness + contrast + sharpness) / 3.0
            quality_scores.append(quality)

        return float(np.mean(quality_scores)) if quality_scores else 0.5

    def process(self, sample: Sample) -> Sample:
        """Process sample with HDR/SDR-aware quality assessment."""
        if not sample.is_video:
            # Process single image
            try:
                img = cv2.imread(str(sample.path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    return sample

                is_hdr = self._detect_hdr(img)

                if is_hdr:
                    quality = self._compute_hdr_quality([img])
                    if sample.quality_metrics is None:
                        sample.quality_metrics = QualityMetrics()
                    sample.quality_metrics.hdr_quality = quality * 100.0
                else:
                    quality = self._compute_sdr_quality([img])
                    if sample.quality_metrics is None:
                        sample.quality_metrics = QualityMetrics()
                    sample.quality_metrics.sdr_quality = quality * 100.0

                logger.debug(f"{'HDR' if is_hdr else 'SDR'} quality for {sample.path.name}: {quality*100:.1f}")

            except Exception as e:
                logger.warning(f"HDR/SDR VQA failed for {sample.path}: {e}")

            return sample

        # Process video
        try:
            cap = cv2.VideoCapture(str(sample.path))
            frames = []
            frame_idx = 0
            is_hdr = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx == 0:
                    is_hdr = self._detect_hdr(frame, video_path=sample.path)

                if frame_idx % self.subsample == 0:
                    frames.append(frame)
                    if len(frames) >= 20:  # Max 20 frames
                        break

                frame_idx += 1

            cap.release()

            if not frames:
                return sample

            if is_hdr:
                quality = self._compute_hdr_quality(frames)
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.hdr_quality = quality * 100.0
            else:
                quality = self._compute_sdr_quality(frames)
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.sdr_quality = quality * 100.0

            logger.debug(f"{'HDR' if is_hdr else 'SDR'} quality for {sample.path.name}: {quality*100:.1f}")

        except Exception as e:
            logger.warning(f"HDR/SDR VQA processing failed for {sample.path}: {e}")

        return sample


class FourKVQAModule(PipelineModule):
    """4K/Ultra-HD video quality assessment with memory-efficient processing."""

    name = "4k_vqa"
    description = "Memory-efficient quality assessment for 4K+ videos"
    default_config = {
        "tile_size": 512,  # Process in tiles to save memory
        "subsample": 10,  # Process fewer frames
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.tile_size = self.config.get("tile_size", 512)
        self.subsample = self.config.get("subsample", 10)

    def setup(self) -> None:
        pass

    def _is_4k_plus(self, width: int, height: int) -> bool:
        """Check if resolution is 4K or higher."""
        return width >= 3840 or height >= 2160

    def _process_tiled(self, frame: np.ndarray) -> float:
        """Process frame in tiles for memory efficiency."""
        h, w = frame.shape[:2]
        tile_scores = []

        # Divide into tiles
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = frame[y:min(y+self.tile_size, h), x:min(x+self.tile_size, w)]

                # Compute quality for tile
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                tile_scores.append(min(sharpness / 1000.0, 1.0))

        return float(np.mean(tile_scores)) if tile_scores else 0.5

    def process(self, sample: Sample) -> Sample:
        """Process 4K+ content with memory-efficient tiling."""
        # Check resolution
        width = sample.width
        height = sample.height

        if width is None or height is None or not self._is_4k_plus(width, height):
            return sample  # Not 4K, skip

        try:
            if not sample.is_video:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    quality = self._process_tiled(img) * 100.0

                    # Store in existing quality field (no specific 4k_quality field)
                    if sample.quality_metrics is None:
                        sample.quality_metrics = QualityMetrics()
                    # Use technical_score as proxy
                    sample.quality_metrics.hdr_technical_score = quality

                    logger.debug(f"4K quality for {sample.path.name}: {quality:.1f}")

                return sample

            # Process video
            cap = cv2.VideoCapture(str(sample.path))
            quality_scores = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.subsample == 0:
                    score = self._process_tiled(frame)
                    quality_scores.append(score)
                    if len(quality_scores) >= 10:  # Max 10 frames
                        break

                frame_idx += 1

            cap.release()

            if quality_scores:
                avg_quality = np.mean(quality_scores) * 100.0

                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.hdr_technical_score = avg_quality

                logger.debug(f"4K quality for {sample.path.name}: {avg_quality:.1f}")

        except Exception as e:
            logger.warning(f"4K VQA processing failed for {sample.path}: {e}")

        return sample
