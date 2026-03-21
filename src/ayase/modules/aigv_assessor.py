"""AIGV-Assessor — AI-generated video quality assessment.

Evaluates AI-generated videos across four quality dimensions: static
quality, temporal smoothness, dynamic degree, and text-video alignment.

Backend tiers:
  1. **AIGV-Assessor model** — real model from HuggingFace/GitHub
     (``wangjiarui153/AIGV-Assessor``) via transformers
  2. **CLIP + heuristic** — CLIP for alignment, OpenCV for other dims
  3. **OpenCV heuristic** — pure OpenCV-based proxy
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGVAssessorModule(PipelineModule):
    name = "aigv_assessor"
    description = "AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback)"
    default_config = {"subsample": 8, "trust_remote_code": True, "model_revision": None}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = "heuristic"
        self._model = None
        self._processor = None
        self._clip_model = None
        self._clip_processor = None
        self._device = None

    def setup(self) -> None:
        # Tier 1: Real AIGV-Assessor model
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "wangjiarui153/AIGV-Assessor"
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._model = AutoModel.from_pretrained(model_name, trust_remote_code=trc, revision=rev).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trc, revision=rev)
            self._device = device
            self._backend = "aigv_assessor"
            self._ml_available = True
            logger.info("AIGV-Assessor loaded real model on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("AIGV-Assessor model unavailable: %s", e)

        # Tier 2: CLIP for alignment + heuristics
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._device = device
            self._backend = "clip_heuristic"
            self._ml_available = True
            logger.info("AIGV-Assessor using CLIP + heuristic on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CLIP unavailable: %s", e)

        # Tier 3: Pure heuristic
        try:
            import cv2
            self._ml_available = True
            self._backend = "heuristic"
            logger.info("AIGV-Assessor ready (heuristic proxy mode)")
        except ImportError:
            logger.warning("AIGV-Assessor unavailable: OpenCV not installed")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            self._compute_dimensions(sample)
        except Exception as e:
            logger.warning("AIGV-Assessor failed: %s", e)
        return sample

    def _compute_dimensions(self, sample: Sample) -> None:
        """Compute 4 AIGV dimensions using the best available backend."""
        if self._backend == "aigv_assessor":
            self._compute_real_model(sample)
        else:
            self._compute_heuristic(sample)

    def _compute_real_model(self, sample: Sample) -> None:
        """Compute dimensions using the real AIGV-Assessor model."""
        import torch
        import cv2
        from PIL import Image

        subsample = self.config.get("subsample", 8)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // subsample)))[:subsample]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        cap.release()

        if not frames:
            return

        try:
            inputs = self._processor(images=frames, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Extract dimension scores from model output
            if hasattr(outputs, "logits"):
                scores = outputs.logits.cpu().numpy().flatten()
            elif isinstance(outputs, dict) and "scores" in outputs:
                scores = np.array(outputs["scores"])
            elif isinstance(outputs, torch.Tensor):
                scores = outputs.cpu().numpy().flatten()
            else:
                logger.warning("AIGV-Assessor: unexpected output type, falling back")
                self._compute_heuristic(sample)
                return

            # Map to dimensions (model may output 4 dimension scores)
            if len(scores) >= 4:
                sample.quality_metrics.aigv_static = float(np.clip(scores[0], 0.0, 1.0))
                sample.quality_metrics.aigv_temporal = float(np.clip(scores[1], 0.0, 1.0))
                sample.quality_metrics.aigv_dynamic = float(np.clip(scores[2], 0.0, 1.0))
                sample.quality_metrics.aigv_alignment = float(np.clip(scores[3], 0.0, 1.0))
            elif len(scores) >= 1:
                sample.quality_metrics.aigv_static = float(np.clip(scores[0], 0.0, 1.0))
        except Exception as e:
            logger.info("AIGV-Assessor model inference failed: %s, falling back", e)
            self._compute_heuristic(sample)

    def _compute_heuristic(self, sample: Sample) -> None:
        """Compute 4 AIGV dimensions using CV-based proxy metrics."""
        import cv2

        subsample = self.config.get("subsample", 8)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // subsample)))[:subsample]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        if len(frames) < 2:
            return

        # 1. Static quality: sharpness + contrast
        sharpness_scores = []
        contrast_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(min(lap / 500.0, 1.0))
            contrast_scores.append(float(gray.std()) / 128.0)
        static = float(np.mean(sharpness_scores) * 0.6 + np.mean(contrast_scores) * 0.4)
        sample.quality_metrics.aigv_static = min(1.0, static)

        # 2. Temporal smoothness: inter-frame consistency
        diffs = []
        for i in range(len(frames) - 1):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i + 1].astype(float)))
            diffs.append(diff)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        smoothness = 1.0 / (1.0 + std_diff / max(mean_diff, 1e-6))
        sample.quality_metrics.aigv_temporal = float(smoothness)

        # 3. Dynamic degree: amount of meaningful motion
        if mean_diff < 2.0:
            dynamic = 0.0
        else:
            dynamic = min(1.0, mean_diff / 30.0)
        sample.quality_metrics.aigv_dynamic = float(dynamic)

        # 4. Text-video alignment: CLIP if available, else use existing score
        if self._backend == "clip_heuristic" and sample.caption and sample.caption.text:
            try:
                self._compute_clip_alignment(sample, frames)
            except Exception:
                if sample.quality_metrics.clip_score is not None:
                    sample.quality_metrics.aigv_alignment = sample.quality_metrics.clip_score
        elif sample.caption and sample.caption.text:
            if sample.quality_metrics.clip_score is not None:
                sample.quality_metrics.aigv_alignment = sample.quality_metrics.clip_score

    def _compute_clip_alignment(self, sample: Sample, frames) -> None:
        """Compute text-video alignment using CLIP."""
        import torch
        import cv2
        from PIL import Image

        text = sample.caption.text
        # Sample a few frames for CLIP
        pil_frames = []
        for f in frames[:4]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        inputs = self._clip_processor(
            text=[text], images=pil_frames, return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            # Image-text similarity (CLIP logits are cosine similarity * 100)
            logits = outputs.logits_per_text  # [1, num_images]
            score = logits[0].mean().item() / 100.0

        sample.quality_metrics.aigv_alignment = float(np.clip(score, 0.0, 1.0))
