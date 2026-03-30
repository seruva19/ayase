import logging
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class ActionRecognitionModule(PipelineModule):
    name = "action_recognition"
    description = "Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models"
    default_config = {
        "model_name": "MCG-NJU/videomae-large-finetuned-kinetics",
        "caption_matching": False,
        "matching_mode": "weighted",  # "weighted" (top-K weighted sum) or "top1" (direct top-1 similarity)
        "clip_model": "openai/clip-vit-base-patch32",
        "top_k": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "MCG-NJU/videomae-large-finetuned-kinetics")
        self.caption_matching = self.config.get("caption_matching", False)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.top_k = self.config.get("top_k", 5)

        self._model = None
        self._processor = None
        self._clip_model = None
        self._clip_tokenizer = None
        self._device = "cpu"
        self._ml_available = False
        self._clip_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading VideoMAE ({self.model_name}) on {self._device}...")

            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

            self._processor = VideoMAEImageProcessor.from_pretrained(resolved)
            self._model = VideoMAEForVideoClassification.from_pretrained(resolved, use_safetensors=True).to(
                self._device
            )
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers not installed. Action recognition disabled.")
        except Exception as e:
            logger.warning(f"Failed to load VideoMAE: {e}")

        if self._ml_available and self.caption_matching:
            self._init_clip()

    def _init_clip(self) -> None:
        self._clip_backend = None  # "open_clip" or "transformers"

        # Tier 1: open_clip (preferred — wider model support)
        try:
            import open_clip

            clip_name = self.config.get("open_clip_model", "ViT-B-32")
            clip_pretrained = self.config.get("open_clip_pretrained", "openai")
            logger.info(f"Loading open_clip ({clip_name}/{clip_pretrained}) for action matching...")
            model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained=clip_pretrained, device=self._device)
            model.eval()
            self._clip_model = model
            self._clip_tokenizer = open_clip.get_tokenizer(clip_name)
            self._clip_available = True
            self._clip_backend = "open_clip"
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"open_clip failed: {e}")

        # Tier 2: transformers CLIPModel
        try:
            from transformers import CLIPModel, AutoTokenizer
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved_clip = resolve_model_path(self.clip_model_name, models_dir)
            logger.info(f"Loading CLIP ({self.clip_model_name}) for action matching on {self._device}...")
            self._clip_model = CLIPModel.from_pretrained(resolved_clip).to(self._device)
            self._clip_tokenizer = AutoTokenizer.from_pretrained(resolved_clip)
            self._clip_available = True
            self._clip_backend = "transformers"
        except ImportError:
            logger.warning("No CLIP backend available. Caption matching disabled.")
        except Exception as e:
            logger.warning(f"Failed to load CLIP for action matching: {e}")

    def _encode_text(self, text):
        """Encode text(s) via CLIP, returns L2-normalized features. Supports both backends."""
        import torch

        if self._clip_backend == "open_clip":
            tokens = self._clip_tokenizer(text if isinstance(text, list) else [text]).to(self._device)
            with torch.no_grad():
                features = self._clip_model.encode_text(tokens)
        else:
            tokens = self._clip_tokenizer(
                text if isinstance(text, list) else [text],
                return_tensors="pt", padding=True, truncation=True,
            )
            with torch.no_grad():
                features = extract_features(self._clip_model.get_text_features(tokens["input_ids"].to(self._device)))

        return features / features.norm(p=2, dim=-1, keepdim=True)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        frames = self._load_frames(sample, num_frames=16)  # VideoMAE usually expects 16 frames
        if len(frames) < 16:
            return sample

        try:
            import torch

            # Prepare inputs
            inputs = self._processor(list(frames), return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = logits.softmax(dim=1)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            # Get top-1 for action_confidence (always available)
            top1_prob, top1_idx = probs.topk(1)
            top1_label = self._model.config.id2label[top1_idx.item()]
            top1_confidence = top1_prob.item()
            sample.quality_metrics.action_confidence = round(top1_confidence * 100.0, 2)

            # Caption-matching mode: compute action_score via CLIP text similarity
            expected_action = self.config.get("expected_action")
            match_text = expected_action or (sample.caption.text if sample.caption else None)

            matching_mode = self.config.get("matching_mode", "weighted")

            if self._clip_available and match_text:
                caption_features = self._encode_text(match_text)

                if matching_mode == "top1":
                    # Direct top-1 similarity: CLIP(top1_label, prompt)
                    action_features = self._encode_text(top1_label)
                    action_score = float((action_features @ caption_features.T).squeeze().cpu()) * 100.0
                else:
                    # Weighted top-K: sum(confidence_i * similarity_i)
                    top_k_probs, top_k_indices = probs.topk(min(self.top_k, probs.shape[1]))
                    action_labels = [
                        self._model.config.id2label[idx.item()]
                        for idx in top_k_indices.squeeze(0)
                    ]
                    confidences = top_k_probs.squeeze(0)
                    action_features = self._encode_text(action_labels)
                    similarities = (action_features @ caption_features.T).squeeze(-1)
                    action_score = (confidences @ similarities).item() * 100.0

                sample.quality_metrics.action_score = round(action_score, 2)
            else:
                # Fallback: use top-1 confidence as action_score (original behavior)
                sample.quality_metrics.action_score = round(top1_confidence * 100.0, 2)

            if top1_confidence > 0.5:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Detected Action: {top1_label} ({top1_confidence:.2f})",
                        details={"action": top1_label, "confidence": top1_confidence},
                    )
                )

                # Add to detections for Knowledge Graph analysis
                sample.detections.append({
                    "type": "action",
                    "label": top1_label,
                    "confidence": float(top1_confidence)
                })

                # Validation against caption
                if sample.caption:
                    if top1_label.replace("_", " ") not in sample.caption.text.lower():
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                message=f"Action '{top1_label}' not explicitly found in caption.",
                            )
                        )

        except Exception as e:
            logger.warning(f"Action recognition failed: {e}")

        return sample

    def _load_frames(self, sample: Sample, num_frames: int = 16) -> List[np.ndarray]:
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                # Loop or duplicate? Better to just take what we have and resample?
                # VideoMAE needs exactly 16 usually for the processor
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()
        except Exception:
            logger.debug("Failed to release video capture for action recognition.")

        # Pad if needed (though we tried to sample 16)
        if len(frames) > 0 and len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1])

        return frames
