import cv2
import numpy as np
import logging
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FrameSampler:
    """
    Unified utility for sampling frames from videos or images.
    Ensures consistent temporal coverage across all modules.
    """

    @staticmethod
    def sample_frames(
        source: Union[str, Path], num_frames: int = 8, uniform: bool = True
    ) -> List[np.ndarray]:
        """
        Samples frames from a video file.

        Args:
            source: Path to the video or image file.
            num_frames: Number of frames to sample.
            uniform: If True, samples uniformly across duration. If False, samples sequentially from start (not recommended for video).

        Returns:
            List of numpy arrays (BGR format).
        """
        source_path = str(source)
        frames = []

        try:
            # Check if likely an image by extension first (optimization)
            if source_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")):
                img = cv2.imread(source_path)
                if img is not None:
                    return [img]

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                # Fallback: maybe it really was an image but weird extension?
                img = cv2.imread(source_path)
                if img is not None:
                    return [img]
                logger.warning(f"Could not open {source_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Helper for single image treated as video
            if total_frames <= 1:
                ret, frame = cap.read()
                cap.release()
                return [frame] if ret else []

            if uniform:
                # Uniform sampling: e.g. 0, 10, 20...
                # Avoid the very last frame if it causes seek issues, but usually fine.
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                indices = np.unique(indices)  # Handle short videos where num_frames > total
            else:
                indices = range(min(num_frames, total_frames))

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    # Retry once after seeking to the same frame.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        logger.debug(f"Failed to read frame {idx} from {source_path}")

            cap.release()

        except Exception as e:
            logger.error(f"Frame sampling failed for {source}: {e}")

        return frames

    @staticmethod
    def load_single_image(source: Union[str, Path]) -> Optional[np.ndarray]:
        """Backward compatibility / Single frame convenience."""
        frames = FrameSampler.sample_frames(source, num_frames=1)
        return frames[0] if frames else None
