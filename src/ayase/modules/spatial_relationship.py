import logging
from typing import List, Dict

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class SpatialRelationshipModule(PipelineModule):
    name = "spatial_relationship"
    description = "Verifies spatial relations (left/right/top/bottom) in prompt vs detections"
    default_config = {}

    def process(self, sample: Sample) -> Sample:
        if not sample.caption or not sample.detections:
            return sample

        caption_lower = sample.caption.text.lower()
        width = sample.width
        height = sample.height
        
        if not width or not height:
            return sample

        # Simple spatial keywords mapping
        # We only check single object spatial constraints for now
        # e.g. "dog on the left" -> find dog, check if x < w/2
        
        # Spatial terms with synonym groups for better recall
        spatial_groups = {
            "left": {
                "keywords": ["left", "on the left", "to the left", "left side", "left of"],
                "check": lambda box: (box[0] + box[2] / 2) < width * 0.4,
            },
            "right": {
                "keywords": ["right", "on the right", "to the right", "right side", "right of"],
                "check": lambda box: (box[0] + box[2] / 2) > width * 0.6,
            },
            "top": {
                "keywords": ["top", "above", "upper", "on top"],
                "check": lambda box: (box[1] + box[3] / 2) < height * 0.4,
            },
            "bottom": {
                "keywords": ["bottom", "below", "lower", "underneath", "beneath"],
                "check": lambda box: (box[1] + box[3] / 2) > height * 0.6,
            },
            "center": {
                "keywords": ["center", "middle", "centered"],
                "check": lambda box: (width * 0.25) < ((box[0] + box[2]) / 2) < (width * 0.75),
            },
            "foreground": {
                "keywords": ["foreground", "in front"],
                "check": lambda box: (box[3] / height) > 0.3,  # large = close = foreground
            },
            "background": {
                "keywords": ["background", "in the back", "behind"],
                "check": lambda box: (box[3] / height) < 0.15,  # small = far = background
            },
        }

        issues = []

        for term_name, group in spatial_groups.items():
            matched_keyword = None
            for kw in group["keywords"]:
                if kw in caption_lower:
                    matched_keyword = kw
                    break
            if not matched_keyword:
                continue

            check_fn = group["check"]
            mentioned_objects = [d for d in sample.detections if d.get("label", "").lower() in caption_lower]
            if not mentioned_objects:
                continue

            matches_spatial = any(check_fn(d["box"]) for d in mentioned_objects if "box" in d)
            if not matches_spatial:
                labels = set(d.get("label", "?") for d in mentioned_objects)
                issues.append(
                    f"Caption mentions '{matched_keyword}' but objects ({labels}) not found in that region."
                )

        if issues:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Spatial mismatch: {'; '.join(issues)}",
                    details={"issues": issues}
                )
            )

        return sample
