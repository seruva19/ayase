import logging
import os
import subprocess
import json
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class AudioModule(PipelineModule):
    name = "audio"
    description = "Validates audio stream quality and presence"
    default_config = {
        "require_audio": False,
        "min_sample_rate": 44100,
        "min_bit_rate": 128000,
        "check_silence": True,
        "silence_threshold": -60.0, # dB
        "emit_audio_tags": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.require_audio = self.config.get("require_audio", False)
        self.min_sample_rate = self.config.get("min_sample_rate", 44100)
        self.min_bit_rate = self.config.get("min_bit_rate", 128000)
        self.check_silence = self.config.get("check_silence", True)
        self.silence_threshold = self.config.get("silence_threshold", -60.0)
        self.emit_audio_tags = self.config.get("emit_audio_tags", True)
        self._ffprobe_available = False
        self._ffmpeg_available = False

    def on_mount(self) -> None:
        super().on_mount()
        import shutil
        self._ffprobe_available = shutil.which("ffprobe") is not None
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        if not self._ffprobe_available:
            logger.warning("ffprobe not found in PATH. Audio validation disabled.")
        if self.check_silence and not self._ffmpeg_available:
            logger.warning("ffmpeg not found in PATH. Silence check disabled.")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if not self._ffprobe_available:
            return sample

        try:
            # Use ffprobe to get audio info
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", str(sample.path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {sample.path}")
                return sample

            probe = json.loads(result.stdout)
            audio_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)

            if not audio_stream:
                if self.require_audio:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="No audio stream found.",
                            recommendation="Add audio or ignore if silent video is expected."
                        )
                    )
                else:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message="No audio stream found (Silent video)."
                        )
                    )
                return sample

            # Validate audio properties
            sample_rate = int(audio_stream.get("sample_rate", 0))
            bit_rate = int(audio_stream.get("bit_rate", 0))
            channels = int(audio_stream.get("channels", 0))

            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Audio detected: {audio_stream.get('codec_name')}, {sample_rate}Hz, {channels}ch",
                    details={"sample_rate": sample_rate, "bit_rate": bit_rate, "channels": channels}
                )
            )

            if sample_rate < self.min_sample_rate:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low audio sample rate: {sample_rate}Hz (Min: {self.min_sample_rate}Hz)",
                    )
                )

            if self.emit_audio_tags:
                tags = ["audio", f"{channels}ch", f"{sample_rate}hz"]
                if bit_rate and bit_rate < self.min_bit_rate:
                    tags.append("low_bitrate")
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Audio tags: {', '.join(tags)}",
                        details={"audio_tags": tags},
                    )
                )

            # Silence Check (using pydub or ffmpeg volumedetect)
            if self.check_silence:
                if not self._ffmpeg_available:
                    return sample
                self._check_audio_energy(sample)

        except Exception as e:
            logger.warning(f"Audio validation failed for {sample.path}: {e}")

        return sample

    def _check_audio_energy(self, sample: Sample):
        try:
            # Use ffmpeg volumedetect filter as it's fast
            cmd = [
                "ffmpeg", "-i", str(sample.path), "-af", "volumedetect",
                "-vn", "-sn", "-dn", "-f", "null", "NUL" if os.name == 'nt' else "/dev/null"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Find 'max_volume' in stderr
            for line in result.stderr.split('\n'):
                if "max_volume:" in line:
                    max_vol = float(line.split("max_volume:")[1].split("dB")[0].strip())
                    if max_vol < self.silence_threshold:
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Audio is very quiet (Max Volume: {max_vol} dB)",
                                details={"max_volume": max_vol, "threshold": self.silence_threshold}
                            )
                        )
                    break
        except Exception as e:
            logger.debug(f"Volume check failed: {e}")
