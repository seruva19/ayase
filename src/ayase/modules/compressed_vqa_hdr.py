"""CompressedVQA-HDR — HDR Compressed Video Quality (ICME 2025 winner).

GitHub: https://github.com/sunwei925/CompressedVQA-HDR

``compressed_vqa_hdr`` is produced only by the real CompressedVQA-HDR model.
When the model/package is not installed the metric is left unset (no
PU/structural-similarity proxy).

compressed_vqa_hdr — higher = better
"""
import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class CompressedVQAHDRModule(ReferenceBasedModule):
    name = "compressed_vqa_hdr"
    description = "CompressedVQA-HDR FR quality (ICME 2025)"
    metric_field = "compressed_vqa_hdr"
    default_config = {"subsample": 8}
    metric_groups = {
        "compressed_vqa_hdr": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import compressedvqa_hdr

            self._model = compressedvqa_hdr
            self._ml_available = True
            self._backend = "compressed_vqa_hdr"
            logger.info("CompressedVQA-HDR initialised (native model)")
        except ImportError:
            logger.warning(
                "CompressedVQA-HDR unavailable: the CompressedVQA-HDR model is not "
                "installed (github.com/sunwei925/CompressedVQA-HDR); metric skipped."
            )

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        if not self._ml_available:
            return None
        try:
            return float(self._model.predict(str(sample_path), str(reference_path)))
        except Exception as e:
            logger.warning(f"CompressedVQA-HDR failed: {e}")
            return None
