"""VMBench Commonsense Adherence Score (CAS) — physical-commonsense plausibility.

Faithful port of VMBench's Commonsense Adherence Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076). A VideoMAEv2 ViT-giant classifier, fine-tuned by VMBench into a
5-level ordinal "commonsense" head, rates how physically plausible the motion is;
the score is the expected value of that ordinal rating:

    CAS = sum( softmax(logits) * [0, 0.25, 0.5, 0.75, 1.0] )     (0-1, higher=better)

The backend is the in-tree vendored VideoMAEv2 (``ayase.vendor.videomae``, weights
``vit_g_vmbench.pt`` from GD-ML/VMBench on HF, pure ``pip install ayase``). This
module produces the single-view estimate (one centre crop of a 16-frame, stride-4
sample); VMBench's published number averages 30 spatio-temporal views.
"""

import logging

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VMBenchCommonsenseAdherenceModule(PipelineModule):
    name = "vmbench_cas"
    description = "VMBench Commonsense Adherence — VideoMAEv2 ordinal plausibility rating (0-1, higher=better)"
    default_config = {
        "device": "auto",
        "max_frames": 64,       # frames handed to the sampler (it selects 16 @ stride 4)
        "models_dir": "models",
    }
    metric_info = {
        "commonsense_adherence_score": "VMBench Commonsense Adherence (VideoMAEv2 ordinal plausibility; 0-1, higher=better)",
    }
    metric_groups = {
        "commonsense_adherence_score": "motion",
    }
    models = [
        {"id": "GD-ML/VMBench", "type": "huggingface",
         "url": "https://huggingface.co/GD-ML/VMBench",
         "task": "VideoMAEv2 ViT-giant commonsense head (vit_g_vmbench.pt)",
         "notes": "Loaded via vendored ayase.vendor.videomae"},
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            device_str = "cuda" if str(device).startswith("cuda") else "cpu"

            from ayase.vendor.videomae import load_videomae

            self._model = load_videomae(
                models_dir=self.config.get("models_dir", "models"), device=device_str)
            self._ml_available = True
            logger.info("VMBench CAS initialised (vendored VideoMAEv2 vit_g backend)")
        except ImportError as e:
            logger.warning("VMBench CAS unavailable (missing backend): %s", e)
        except Exception as e:
            logger.warning("Failed to setup VMBench CAS: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample
        try:
            frames = sample_frames(sample.path, max_frames=int(self.config.get("max_frames", 64)),
                                   color="rgb")
            if len(frames) < 4:
                return sample
            cas = float(self._model.commonsense_adherence_score(frames))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.commonsense_adherence_score = cas
            logger.debug("VMBench CAS for %s: %.4f", sample.path.name, cas)
        except Exception as e:
            logger.warning("VMBench CAS processing failed for %s: %s", sample.path, e)
        return sample
