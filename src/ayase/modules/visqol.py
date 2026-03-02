"""ViSQOL (Virtual Speech Quality Objective Listener) module.

ViSQOL is Google's full-reference audio quality metric that predicts
MOS-LQO (Mean Opinion Score - Listening Quality Objective) using
spectro-temporal similarity analysis.

Range: 1-5 MOS (higher = better).

Requires the ``visqol`` Python package or CLI.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ViSQOLModule(PipelineModule):
    name = "visqol"
    description = "ViSQOL audio quality MOS (Google, 1-5, higher=better)"
    default_config = {
        "mode": "audio",  # "audio" or "speech"
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.mode = self.config.get("mode", "audio")
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try Python bindings
        try:
            from visqol import visqol_lib_py
            self._backend = "python"
            self._ml_available = True
            logger.info("ViSQOL module initialised (Python bindings)")
            return
        except ImportError:
            pass

        # Try CLI
        try:
            result = subprocess.run(
                ["visqol", "--help"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 or "visqol" in result.stdout.lower():
                self._backend = "cli"
                self._ml_available = True
                logger.info("ViSQOL module initialised (CLI)")
                return
        except FileNotFoundError:
            pass
        except Exception:
            pass

        logger.warning("ViSQOL not available. Install from: https://github.com/google/visqol")

    def _compute_visqol_python(
        self, reference_path: str, degraded_path: str
    ) -> Optional[float]:
        try:
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2, similarity_result_pb2

            config = visqol_config_pb2.VisqolConfig()
            if self.mode == "speech":
                config.audio.sample_rate = 16000
                config.options.use_speech_scoring = True
                svr_model = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
            else:
                config.audio.sample_rate = 48000
                config.options.use_speech_scoring = False
                svr_model = "libsvm_nu_svr_model.txt"
            config.options.svr_model_path = svr_model

            api = visqol_lib_py.VisqolApi()
            api.Create(config)
            result = api.Measure(reference_path, degraded_path)
            return float(result.moslqo)
        except Exception as e:
            logger.debug(f"ViSQOL Python scoring failed: {e}")
            return None

    def _compute_visqol_cli(
        self, reference_path: str, degraded_path: str
    ) -> Optional[float]:
        try:
            cmd = [
                "visqol",
                "--reference_file", reference_path,
                "--degraded_file", degraded_path,
                "--use_speech_scoring" if self.mode == "speech" else "",
            ]
            cmd = [c for c in cmd if c]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            import re
            match = re.search(r"MOS-LQO:\s+([0-9.]+)", result.stdout)
            if match:
                return float(match.group(1))
            return None
        except Exception as e:
            logger.debug(f"ViSQOL CLI failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            if self._backend == "python":
                score = self._compute_visqol_python(str(reference), str(sample.path))
            else:
                score = self._compute_visqol_cli(str(reference), str(sample.path))

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.visqol = score
            logger.debug(f"ViSQOL for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"ViSQOL failed: {e}")
        return sample
