"""Perceptual duplicate detection using pHash (perceptual hashing) across the dataset.

Computes a perceptual hash of the middle frame and flags exact hash matches.
Cross-sample deduplication requires processing the full dataset."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Set, Dict, List, Optional

from ayase.image import load_representative_frame
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

    def setup(self) -> None:
        self.seen_hashes = {}

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
                self.seen_hashes[phash] = str(sample.path)
                
        except Exception as e:
            logger.warning(f"Dedup failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            # Middle frame for video, the image itself otherwise (shared cache).
            return load_representative_frame(sample.path, color="bgr")
        except Exception:
            return None


class DedupCompatModule(DeduplicationModule):
    """Compatibility alias matching filename-based discovery."""

    name = "dedup"
