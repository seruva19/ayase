import logging
import numpy as np
import cv2
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoTextMatchingModule(PipelineModule):
    name = "video_text_matching"
    description = "ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP"
    default_config = {
        "use_xclip": False, # Use heavy video-native model
        "model_name": "openai/clip-vit-base-patch32",
        "xclip_model_name": "microsoft/xclip-base-patch32",
        "min_score_threshold": 0.20,
        "consistency_std_threshold": 0.1,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_xclip = self.config.get("use_xclip", False)
        
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.xclip_model_name = self.config.get("xclip_model_name", "microsoft/xclip-base-patch32")
        
        self.min_score_threshold = self.config.get("min_score_threshold", 0.20)
        self.consistency_std_threshold = self.config.get("consistency_std_threshold", 0.1)
        
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._is_xclip = False

    def on_mount(self) -> None:
        super().on_mount()
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up Video-Text Matching on {self._device}...")

            from transformers import CLIPModel, CLIPProcessor, XCLIPModel, XCLIPProcessor
            models_dir = self.config.get("models_dir", "models")

            if self.use_xclip:
                try:
                    logger.info(f"Loading X-CLIP ({self.xclip_model_name})...")
                    self._model = XCLIPModel.from_pretrained(
                        self.xclip_model_name, cache_dir=models_dir
                    ).to(self._device)
                    self._processor = XCLIPProcessor.from_pretrained(self.xclip_model_name, cache_dir=models_dir)
                    self._is_xclip = True
                    self._ml_available = True
                    return
                except Exception as e:
                    logger.warning(f"Failed to load X-CLIP: {e}. Falling back to standard CLIP.")
                    self._is_xclip = False

            # Fallback to standard CLIP
            logger.info(f"Loading CLIP ({self.model_name})...")
            self._model = CLIPModel.from_pretrained(
                self.model_name, cache_dir=models_dir
            ).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=models_dir)

            self._ml_available = True

        except ImportError:
            logger.warning("Transformers not installed.")
        except Exception as e:
            logger.warning(f"Failed to setup VideoTextMatching: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.caption:
            return sample

        try:
            if self._is_xclip:
                self._process_xclip(sample)
            else:
                self._process_clip(sample)

        except Exception as e:
            logger.warning(f"Video-Text Matching check failed: {e}")

        return sample

    def _process_xclip(self, sample: Sample):
        import torch
        frames = self._load_frames(sample, num_frames=8)
        if not frames:
            return

        from PIL import Image

        text = sample.caption.text[:77]

        # XCLIPProcessor expects videos as a list of list of PIL images
        pil_frames = [Image.fromarray(f) for f in frames]
        inputs = self._processor(
            text=[text], videos=[pil_frames], return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_video  # [B_video, B_text]
            # X-CLIP logits are already cosine similarities scaled by temperature
            score = logits[0][0].item()
            # Normalize to approximate 0-1 range (logits are typically ~0-30)
            normalized = score / logits.abs().max().clamp(min=1.0).item()

            if not sample.quality_metrics:
                from ayase.models import QualityMetrics
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.clip_score = float(normalized)
            sample.quality_metrics.clip_temp = 1.0

            if normalized < self.min_score_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Video-Text Alignment (X-CLIP): {normalized:.2f}",
                        details={"xclip_score": float(score), "xclip_normalized": float(normalized)},
                    )
                )

    def _process_clip(self, sample: Sample):
            import torch
            frames = self._load_frames(sample, num_frames=5)
            if not frames:
                return

            from PIL import Image

            pil_images = [Image.fromarray(f) for f in frames]
            text = sample.caption.text[:77]

            inputs = self._processor(
                text=[text], images=pil_images, return_tensors="pt", padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # logits_per_image: [num_frames, 1]
                logits = outputs.logits_per_image
                # Use raw cosine similarity (normalized)
                raw_scores = logits / 100.0

                avg_score = torch.mean(raw_scores).item()
                score_std = torch.std(raw_scores).item()

                if not sample.quality_metrics:
                    from ayase.models import QualityMetrics
                    sample.quality_metrics = QualityMetrics()

                sample.quality_metrics.clip_score = avg_score
                sample.quality_metrics.clip_temp = max(0.0, 1.0 - score_std)

                if avg_score < self.min_score_threshold:
                     sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Low Video-Text Alignment (Avg CLIP): {avg_score:.2f}",
                            details={"avg_clip_score": avg_score},
                            recommendation="Rewrite caption or discard."
                        )
                    )
                
                if score_std > self.consistency_std_threshold:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Inconsistent Text Match (StdDev: {score_std:.2f})",
                            details={"score_std": score_std},
                        )
                    )

    def _load_frames(self, sample: Sample, num_frames: int = 5) -> list:
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return []
            
            indices = np.linspace(0, total - 1, num_frames, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
        except Exception:
            logger.debug(f"Failed to load frames for {sample.path}")
        return frames

