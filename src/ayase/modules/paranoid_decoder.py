import logging
import subprocess
import os
import shutil
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class ParanoidDecoderModule(PipelineModule):
    """
    Performs deep bitstream validation by attempting to decode the entire file.
    This detects corruption that simple metadata checks (MOOV atom etc.) might miss.
    """
    name = "paranoid_decoder"
    description = "Deep bitstream validation using FFmpeg (Paranoid Mode)"
    default_config = {
        "timeout": 60,  # Max seconds to wait for decoding
        "strict_mode": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.timeout = self.config.get("timeout", 60)
        self.strict_mode = self.config.get("strict_mode", True)
        self._ffmpeg_available = False

    def setup(self) -> None:
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        if not self._ffmpeg_available:
            logger.warning("ffmpeg not found. ParanoidDecoderModule disabled.")

    def process(self, sample: Sample) -> Sample:
        if not self._ffmpeg_available or not sample.is_video:
            return sample

        try:
            # ffmpeg -v error -xerror -i input -f null -
            # -xerror makes it exit with non-zero on any error
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", str(sample.path),
                "-f", "null", "-"
            ]
            
            if self.strict_mode:
                cmd.insert(2, "-xerror")

            # Run with timeout to prevent hanging on severely broken files
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )

            if result.returncode != 0:
                errors = result.stderr.strip()
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Bitstream corruption detected during full decode.",
                        details={"ffmpeg_error": errors},
                        recommendation="The file is corrupted at the codec level and may cause data loader crashes. Discard or re-encode."
                    )
                )
            else:
                logger.debug(f"Paranoid decode passed for {sample.path}")

        except subprocess.TimeoutExpired:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Decoding timed out after {self.timeout}s. File might be extremely complex or semi-corrupted.",
                    details={"timeout": self.timeout}
                )
            )
        except Exception as e:
            logger.warning(f"Paranoid decoder failed for {sample.path}: {e}")

        return sample
