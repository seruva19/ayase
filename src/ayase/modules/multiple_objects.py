"""Multiple Objects module — VBench dimension #9.

Evaluates whether the correct NUMBER of objects appear when the caption
specifies a quantity (e.g. "two dogs", "three cars").  Uses the object
detection results already produced by ObjectDetectionModule.
"""

import logging
import re
from typing import Dict, List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

WORD_TO_NUM = {
    "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "couple": 2, "pair": 2,
    "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "several": -1, "many": -1, "multiple": -1, "few": -1,
}

# Pattern: "<number/word> <optional-adjective(s)> <noun>"
_COUNT_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(w) for w in WORD_TO_NUM)
    + r"|\d+)\s+(?:\w+\s+){0,2}(\w+s)\b",
    re.IGNORECASE,
)


class MultipleObjectsModule(PipelineModule):
    name = "multiple_objects"
    description = "Verifies object count matches caption (VBench multiple_objects dimension)"
    default_config = {
        "tolerance": 1,  # allowed count mismatch before flagging
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.tolerance = self.config.get("tolerance", 1)

    def process(self, sample: Sample) -> Sample:
        if not sample.caption or not sample.detections:
            return sample

        caption = sample.caption.text.lower()

        # Extract expected counts from caption
        expected = self._parse_counts(caption)
        if not expected:
            return sample

        # Count detected objects by label
        detected_counts: Dict[str, int] = {}
        for det in sample.detections:
            label = det.get("label", "").lower()
            if label:
                detected_counts[label] = detected_counts.get(label, 0) + 1

        # Compare expected vs detected
        for noun, expected_count in expected.items():
            if expected_count == -1:
                # Vague quantifier ("several", "many") — just check > 1
                total = self._find_matching_count(noun, detected_counts)
                if total <= 1:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Caption says multiple '{noun}' but only {total} detected",
                            details={"noun": noun, "expected": "multiple", "detected": total},
                        )
                    )
                continue

            total = self._find_matching_count(noun, detected_counts)
            diff = abs(total - expected_count)
            if diff > self.tolerance:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Object count mismatch: caption says {expected_count} '{noun}', detected {total}",
                        details={
                            "noun": noun,
                            "expected": expected_count,
                            "detected": total,
                        },
                        recommendation="Caption quantity does not match visual content; re-caption or verify detections.",
                    )
                )

        return sample

    def _parse_counts(self, caption: str) -> Dict[str, int]:
        """Extract (noun -> expected count) from caption text."""
        counts: Dict[str, int] = {}
        for match in _COUNT_PATTERN.finditer(caption):
            raw_num = match.group(1).lower()
            noun = match.group(2).lower()

            if raw_num in WORD_TO_NUM:
                num = WORD_TO_NUM[raw_num]
            else:
                try:
                    num = int(raw_num)
                except ValueError:
                    continue

            # Only track if asking for >0
            if num != 0:
                counts[noun] = num
        return counts

    @staticmethod
    def _find_matching_count(noun: str, detected: Dict[str, int]) -> int:
        """Find detections matching *noun* (singular/plural fuzzy)."""
        singular = noun.rstrip("s")
        total = 0
        for label, count in detected.items():
            if label == noun or label == singular or label.rstrip("s") == singular:
                total += count
        return total
