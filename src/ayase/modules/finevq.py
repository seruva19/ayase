"""FineVQ — fine-grained video quality assessment.

Decomposes video quality into sub-dimensions: sharpness, colorfulness,
noise, temporal stability, and content richness. Fuses sub-scores via
configurable weights.

Backend tiers:
  1. **FineVQ model** — real FineVQ model from HuggingFace
     (``IntMeGroup/FineVQ_score``)
  2. **TOPIQ + handcrafted** — pyiqa TOPIQ-NR backbone + OpenCV features
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FineVQModule(PipelineModule):
    name = "finevq"
    description = "Fine-grained video quality (FineVQ model or TOPIQ+handcrafted)"
    default_config = {
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
        "weights": {
            "sharpness": 0.20,
            "colorfulness": 0.15,
            "noise": 0.20,
            "temporal_stability": 0.25,
            "content_richness": 0.20,
        },
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._topiq = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real FineVQ model from HuggingFace
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "IntMeGroup/FineVQ_score"
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(model_name, trust_remote_code=trc, revision=rev).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trc, revision=rev)
            self._device = device
            self._backend = "finevq"
            self._ml_available = True
            logger.info("FineVQ loaded real model from HuggingFace on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("FineVQ model unavailable: %s", e)

        # Tier 2: TOPIQ backbone + handcrafted features
        try:
            import torch
            import pyiqa

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._topiq = pyiqa.create_metric("topiq_nr", device=device)
            self._device = device
            self._backend = "topiq_handcrafted"
            self._ml_available = True
            logger.info("FineVQ initialised (TOPIQ backbone) on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.warning("FineVQ: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            if self._backend == "finevq":
                self._process_real_model(sample)
            elif self._backend == "topiq_handcrafted":
                self._process_handcrafted(sample)
        except Exception as e:
            logger.warning("FineVQ failed: %s", e)
        return sample

    def _process_real_model(self, sample: Sample) -> None:
        """Process using the real FineVQ model."""
        import torch
        import cv2
        from PIL import Image

        frames_pil = []
        frames_cv = self._load_frames(sample)
        if not frames_cv:
            return

        for f in frames_cv:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))

        try:
            inputs = self._processor(images=frames_pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            if hasattr(outputs, "logits"):
                score = outputs.logits.mean().item()
            elif isinstance(outputs, torch.Tensor):
                score = outputs.mean().item()
            elif isinstance(outputs, dict) and "score" in outputs:
                score = float(outputs["score"])
            else:
                score = None

            if score is not None:
                # Don't clamp: real FineVQ model may output scores outside [0,1]
                sample.quality_metrics.finevq_score = float(score)
        except Exception as e:
            logger.warning("FineVQ model inference failed: %s", e)

    def _process_handcrafted(self, sample: Sample) -> None:
        """Process using handcrafted features (+optional TOPIQ)."""
        import cv2

        frames = self._load_frames(sample)
        if not frames:
            return

        # Sub-dimension 1: Sharpness (Laplacian variance)
        sharpness_scores = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(min(1.0, lap_var / 500.0))

        # Sub-dimension 2: Colorfulness (Hasler & Susstrunk)
        color_scores = []
        for f in frames:
            b, g, r = f[:, :, 0].astype(float), f[:, :, 1].astype(float), f[:, :, 2].astype(float)
            rg = r - g
            yb = 0.5 * (r + g) - b
            std_rg, std_yb = np.std(rg), np.std(yb)
            mean_rg, mean_yb = np.mean(rg), np.mean(yb)
            colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
            color_scores.append(min(1.0, colorfulness / 150.0))

        # Sub-dimension 3: Noise level
        noise_scores = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sigma = np.median(np.abs(lap)) * 1.4826
            noise_scores.append(max(0.0, 1.0 - sigma / 30.0))

        # Sub-dimension 4: Temporal stability
        temporal_scores = []
        if len(frames) >= 2:
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                mse = np.mean((g1 - g2) ** 2)
                temporal_scores.append(1.0 / (1.0 + mse / 500.0))
        else:
            temporal_scores = [1.0]

        # Sub-dimension 5: Content richness (spatial entropy)
        richness_scores = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            richness_scores.append(min(1.0, entropy / 8.0))

        # Optional neural boost from TOPIQ
        neural_boost = 0.0
        if self._topiq is not None:
            try:
                import torch
                nn_scores = []
                for f in frames[:4]:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(self._device)
                    with torch.no_grad():
                        s = self._topiq(tensor).item()
                    nn_scores.append(s)
                neural_boost = float(np.mean(nn_scores))
            except Exception:
                pass

        weights = self.config.get("weights", self.default_config["weights"])
        sub_scores = {
            "sharpness": float(np.mean(sharpness_scores)),
            "colorfulness": float(np.mean(color_scores)),
            "noise": float(np.mean(noise_scores)),
            "temporal_stability": float(np.mean(temporal_scores)),
            "content_richness": float(np.mean(richness_scores)),
        }

        total = sum(weights.get(k, 0.2) * v for k, v in sub_scores.items())
        total_w = sum(weights.get(k, 0.2) for k in sub_scores)
        score = total / max(total_w, 1e-8)

        if neural_boost > 0:
            score = 0.7 * score + 0.3 * neural_boost

        sample.quality_metrics.finevq_score = float(np.clip(score, 0.0, 1.0))

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames
