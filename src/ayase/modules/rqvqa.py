"""RQ-VQA — rich quality-aware video quality assessment.

Multi-attribute VQA evaluating clarity, aesthetics, motion naturalness,
semantic coherence, and overall impression.

Backend tiers:
  1. **RQ-VQA model** — real model from GitHub
     (weights from ``AkaneTendo25/ayase-models``, original: ``sunwei925/RQ-VQA``)
  2. **CLIP-IQA+ backbone** — pyiqa CLIP-IQA+ with handcrafted features
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RQVQAModule(PipelineModule):
    name = "rqvqa"
    description = "Multi-attribute video quality (RQ-VQA model or CLIP-IQA+)"
    default_config = {
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
        "dimensions": {
            "clarity": 0.25,
            "aesthetics": 0.20,
            "motion_naturalness": 0.25,
            "semantic_coherence": 0.15,
            "overall_impression": 0.15,
        },
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._clipiqa = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real RQ-VQA model (custom weights)
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Try loading from a known HuggingFace or local path
            from transformers import AutoModel
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(
                "AkaneTendo25/ayase-models", subfolder="rqvqa",
                trust_remote_code=trc, revision=rev
            ).to(device).eval()
            self._device = device
            self._backend = "rqvqa"
            self._ml_available = True
            logger.info("RQ-VQA loaded real model on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("RQ-VQA model unavailable: %s", e)

        # Tier 2: CLIP-IQA+ backbone
        try:
            import torch
            import pyiqa

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._clipiqa = pyiqa.create_metric("clipiqa+", device=device)
            self._device = device
            self._backend = "clipiqa"
            self._ml_available = True
            logger.info("RQ-VQA using CLIP-IQA+ backbone on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.warning("RQ-VQA: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            # Dispatch to real RQ-VQA model when available
            if self._backend == "rqvqa" and self._model is not None:
                score = self._process_rqvqa_model(sample, frames)
                if score is not None:
                    sample.quality_metrics.rqvqa_score = float(np.clip(score, 0.0, 1.0))
                return sample

            # CLIP-IQA+ backbone
            if self._backend == "clipiqa" and self._clipiqa is not None:
                import torch

                dims = self.config.get("dimensions", self.default_config["dimensions"])

                # Clarity: Laplacian sharpness + contrast
                clarity_scores = []
                for f in frames:
                    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                    contrast = gray.std() / 128.0
                    clarity_scores.append(min(1.0, (lap / 500.0 + contrast) / 2.0))

                # Aesthetics: Color harmony + rule of thirds
                aes_scores = []
                for f in frames:
                    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
                    h_hist = h_hist / (h_hist.sum() + 1e-8)
                    hue_entropy = -np.sum(h_hist[h_hist > 0] * np.log2(h_hist[h_hist > 0]))
                    aes = 1.0 - abs(hue_entropy - 2.5) / 3.5
                    aes_scores.append(max(0.0, min(1.0, aes)))

                # Motion naturalness: optical flow smoothness
                motion_scores = []
                if len(frames) >= 2 and sample.is_video:
                    for i in range(len(frames) - 1):
                        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                        smoothness = 1.0 / (1.0 + np.std(mag) / 5.0)
                        mag_score = float(np.exp(-0.5 * ((np.mean(mag) - 5.0) / 8.0) ** 2))
                        motion_scores.append(0.5 * smoothness + 0.5 * mag_score)
                else:
                    motion_scores = [0.8]

                # Semantic coherence: inter-frame histogram correlation
                coherence_scores = []
                if len(frames) >= 2:
                    prev_hist = None
                    for f in frames:
                        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
                        hist = hist / (hist.sum() + 1e-8)
                        if prev_hist is not None:
                            corr = float(np.corrcoef(hist, prev_hist)[0, 1])
                            coherence_scores.append(max(0.0, corr))
                        prev_hist = hist
                if not coherence_scores:
                    coherence_scores = [1.0]

                # Overall impression: CLIP-IQA+ scoring
                impression_scores = []
                try:
                    for f in frames[:4]:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        tensor = tensor.to(self._device)
                        with torch.no_grad():
                            s = self._clipiqa(tensor).item()
                        impression_scores.append(s)
                except Exception:
                    pass
                if not impression_scores:
                    impression_scores = [
                        np.mean([np.mean(clarity_scores), np.mean(aes_scores),
                                 np.mean(motion_scores), np.mean(coherence_scores)])
                    ]

                sub = {
                    "clarity": float(np.mean(clarity_scores)),
                    "aesthetics": float(np.mean(aes_scores)),
                    "motion_naturalness": float(np.mean(motion_scores)),
                    "semantic_coherence": float(np.mean(coherence_scores)),
                    "overall_impression": float(np.mean(impression_scores)),
                }

                total = sum(dims.get(k, 0.2) * v for k, v in sub.items())
                total_w = sum(dims.get(k, 0.2) for k in sub)
                score = total / max(total_w, 1e-8)

                sample.quality_metrics.rqvqa_score = float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.warning("RQ-VQA failed: %s", e)
        return sample

    def _process_rqvqa_model(self, sample: Sample, frames: list) -> Optional[float]:
        """Process using the real RQ-VQA model."""
        import torch
        import cv2

        try:
            tensors = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensors.append(t)
            clip = torch.stack(tensors).unsqueeze(0).to(self._device)
            with torch.no_grad():
                output = self._model(clip)
                if isinstance(output, dict):
                    score = output.get("score", output.get("quality"))
                elif isinstance(output, (tuple, list)):
                    score = output[0]
                else:
                    score = output
                if hasattr(score, "item"):
                    score = score.item()
            return float(score)
        except Exception as e:
            logger.warning("RQ-VQA real model inference failed, falling back: %s", e)
            return None

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
