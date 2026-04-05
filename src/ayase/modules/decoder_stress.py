"""Random-access decoder stress test for video seek table and bitstream integrity.

Probes deterministic random frame positions via cv2 seek to verify the video
can be decoded at arbitrary offsets. Detects corrupted seek tables and black frames."""

import logging
import cv2
import random
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class DecoderStressModule(PipelineModule):
    """
    Performs a 'Stress Test' by attempting to decode frames at random positions.
    This verifies the integrity of the seek table and B-frame reconstruction.
    """
    name = "decoder_stress"
    description = "Random access decoder stress test"
    default_config = {
        "num_probes": 5,      # Number of random positions to check
        "check_integrity": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_probes = self.config.get("num_probes", 5)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return sample

            # Define probe points: Start, Middle, End, and some deterministic random ones
            probes = [0, total_frames // 2, total_frames - 1]
            if self.num_probes > 3:
                rng = random.Random(hash(str(sample.path)))
                for _ in range(self.num_probes - 3):
                    probes.append(rng.randint(0, total_frames - 1))
            
            # Sort to minimize seeking (though not strictly necessary)
            probes = sorted(list(set(probes)))

            failed_probes = []
            for p in probes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    failed_probes.append(p)
                elif frame.mean() < 1e-3 and p not in (0, total_frames - 1):
                    # Suspicious all-black frame at a non-boundary position
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Suspicious black frame at index {p}",
                            details={"frame_index": p, "mean_value": float(frame.mean())},
                        )
                    )

            cap.release()

            if failed_probes:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Decoder stress test failed at {len(failed_probes)} indices.",
                        details={"failed_indices": failed_probes, "total_probes": len(probes)},
                        recommendation="The video container has seeking errors or corrupted segments. Training might crash when accessing these frames. Discard or re-index the file."
                    )
                )
            else:
                logger.debug(f"Decoder stress test passed for {sample.path}")

        except Exception as e:
            logger.warning(f"Decoder stress test failed for {sample.path}: {e}")

        return sample
