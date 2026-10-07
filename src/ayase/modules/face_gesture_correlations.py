"""Face+gesture correlations — 496-dim behavioural signature (external backend).

Bohacek & Farid (PNAS 2022, arXiv:2206.12043) protect persons of interest by
correlating facial expressions with hand gestures: a 496-dimensional
correlation signature over face and body features separates genuine speakers
from manipulated video.

The paper describes the procedure without releasing code or the annotation
pipeline, so the module stays registered but marked
``requires_external_backend`` until the feature+correlation pipeline is wired
in.

face_gesture_correlation -- authenticity score of the face/gesture signature (higher=more genuine).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceGestureCorrelationsModule(PipelineModule):
    name = "face_gesture_correlations"
    description = "Face/gesture correlation signature (Bohacek & Farid 2022, external backend)"
    provenance = "published"
    sources = {
        "face_gesture_correlation": "496-dim face+gesture correlations, Bohacek & Farid, PNAS 2022 (https://arxiv.org/abs/2206.12043)",
    }
    requires_external_backend = True  # procedure only — no published code
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "face_gesture_correlation": "Face/gesture signature authenticity vs reference person (higher=genuine)",
    }
    metric_groups = {"face_gesture_correlation": "face"}

    def setup(self) -> None:
        logger.warning(
            "face_gesture_correlations unavailable: the source describes the "
            "procedure but ships no code; face_gesture_correlation left unset."
        )

    def process(self, sample):
        return sample
