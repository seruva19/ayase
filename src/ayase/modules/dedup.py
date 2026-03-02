import logging
import cv2
import numpy as np
from PIL import Image
from typing import Set, Dict, List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class DeduplicationModule(PipelineModule):
    name = "deduplication"
    description = "Detects duplicates using Perceptual Hashing (pHash)"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self.seen_hashes: Dict[str, str] = {} # hash -> filename
        self._imagehash_available = False
        
        try:
            import imagehash
            self.imagehash = imagehash
            self._imagehash_available = True
        except ImportError:
            logger.warning("imagehash not installed. Deduplication disabled.")

    def process(self, sample: Sample) -> Sample:
        if not self._imagehash_available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            phash = str(self.imagehash.phash(pil_image))
            
            if phash in self.seen_hashes:
                original_file = self.seen_hashes[phash]
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Duplicate content detected. Similar to: {original_file}",
                        details={"phash": phash, "original": original_file}
                    )
                )
            else:
                self.seen_hashes[phash] = sample.path.name
                
        except Exception as e:
            logger.warning(f"Dedup failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                # Use middle frame for hash
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None


class DedupCompatModule(DeduplicationModule):
    """Compatibility alias matching filename-based discovery."""

    name = "dedup"
