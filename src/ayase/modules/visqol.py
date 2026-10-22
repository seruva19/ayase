"""ViSQOL (Virtual Speech Quality Objective Listener) module.

ViSQOL is Google's full-reference audio quality metric that predicts
MOS-LQO (Mean Opinion Score - Listening Quality Objective) using
spectro-temporal similarity analysis.

Range: 1-5 MOS (higher = better).

The default managed backend uses ``visqol-python`` with the published SVR and
deep-lattice models. Google's native Python binding and CLI remain expert options.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ViSQOLModule(PipelineModule):
    name = "visqol"
    provenance = "published"
    sources = {
        "visqol": "ViSQOL v3, Google — https://github.com/google/visqol",
    }
    description = "ViSQOL audio quality MOS (Google, 1-5, higher=better)"
    models = [
        {
            "id": "visqol-python==3.8.0",
            "type": "pip_package",
            "install": "pip install ayase",
            "url": "https://github.com/talker93/visqol-python",
            "task": "Bundled ViSQOL audio SVR and speech deep-lattice mapping models",
            "notes": (
                "Ayase-managed numerical transcription of the published protocol; "
                "backend identity visqol_python_port; 12 official conformance cases verified"
            ),
        },
    ]
    default_config = {
        "mode": "audio",  # "audio" or "speech"
        "visqol_executable": "visqol",
    }
    metric_groups = {
        "visqol": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.mode = self.config.get("mode", "audio")
        self._ml_available = False
        self._backend = None
        self._visqol_api = None

    def setup(self) -> None:
        # Cross-platform transcription of the published ViSQOL v3 protocol.
        try:
            from visqol import VisqolApi

            api = VisqolApi()
            if self.mode == "speech":
                # Never accept the package's polynomial fallback: Ayase speech
                # scores must use the C++-default deep-lattice mapper.
                api.create(mode="speech", use_lattice_model=True)
            elif self.mode == "audio":
                api.create(mode="audio")
            else:
                raise ValueError(f"Unsupported ViSQOL mode: {self.mode!r}")
            self._visqol_api = api
            self._backend = "visqol_python_port"
            self._ml_available = True
            logger.info("ViSQOL module initialised (visqol-python port)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning("visqol-python backend unavailable: %s", e)

        # Google's native Python bindings, retained as an expert backend.
        try:
            from visqol import visqol_lib_py

            self._backend = "visqol_python_official"
            self._ml_available = True
            logger.info("ViSQOL module initialised (official Python bindings)")
            return
        except ImportError:
            pass

        # Try CLI
        try:
            result = subprocess.run(
                [self.config.get("visqol_executable", "visqol"), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 or "visqol" in result.stdout.lower():
                self._backend = "visqol_cli"
                self._ml_available = True
                logger.info("ViSQOL module initialised (CLI)")
                return
        except FileNotFoundError:
            pass
        except Exception:
            pass

        self._backend = "unavailable"
        logger.warning("ViSQOL managed backend could not load; see the backend installation log")

    def _compute_visqol_port(self, reference_path: str, degraded_path: str) -> Optional[float]:
        try:
            if self._visqol_api is None:
                return None
            result = self._visqol_api.measure(reference_path, degraded_path)
            return float(result.moslqo)
        except Exception as e:
            logger.warning("visqol-python scoring failed: %s", e)
            return None

    def _compute_visqol_official(
        self, reference_path: str, degraded_path: str
    ) -> Optional[float]:
        try:
            import numpy as np
            import soundfile as sf
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2

            config = visqol_config_pb2.VisqolConfig()
            if self.mode == "speech":
                config.audio.sample_rate = 16000
                config.options.use_speech_scoring = True
                svr_model = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
            else:
                config.audio.sample_rate = 48000
                config.options.use_speech_scoring = False
                svr_model = "libsvm_nu_svr_model.txt"
            config.options.svr_model_path = str(
                Path(visqol_lib_py.__file__).resolve().parent / "model" / svr_model
            )

            reference, reference_rate = sf.read(reference_path, dtype="float64")
            degraded, degraded_rate = sf.read(degraded_path, dtype="float64")
            if reference_rate != config.audio.sample_rate or degraded_rate != config.audio.sample_rate:
                raise ValueError(
                    f"ViSQOL {self.mode} mode requires {config.audio.sample_rate} Hz inputs"
                )
            if reference.ndim > 1:
                reference = np.mean(reference, axis=1)
            if degraded.ndim > 1:
                degraded = np.mean(degraded, axis=1)

            api = visqol_lib_py.VisqolApi()
            api.Create(config)
            result = api.Measure(reference, degraded)
            return float(result.moslqo)
        except Exception as e:
            logger.debug(f"ViSQOL Python scoring failed: {e}")
            return None

    def _compute_visqol_cli(
        self, reference_path: str, degraded_path: str
    ) -> Optional[float]:
        try:
            cmd = [
                self.config.get("visqol_executable", "visqol"),
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
            if self._backend == "visqol_python_port":
                score = self._compute_visqol_port(str(reference), str(sample.path))
            elif self._backend == "visqol_python_official":
                score = self._compute_visqol_official(str(reference), str(sample.path))
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
