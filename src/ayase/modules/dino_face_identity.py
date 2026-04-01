"""DINO Face Identity — reference-based identity metric using DINOv2 face crops.

Better than ArcFace for AI-generated faces: ArcFace systematically underestimates
identity in generated content, while DINOv2 on face crops correlates with human
judgment significantly better. Applicable to all identity-preserving generation
(LoRA, DreamBooth, IP-Adapter, InstantID).

Outputs:
    dino_face_identity     — cosine similarity 0-1 (higher = better match)
    dino_face_identity_max — max similarity across sampled frames

Requires ``sample.reference_path`` pointing to a reference face image or directory.
Gracefully skips when no reference is provided.

Backends:
    - Face detection: InsightFace (buffalo_l)
    - Embedding: DINOv2 ViT-B/14 (facebookresearch/dinov2)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DINOFaceIdentityModule(PipelineModule):
    name = "dino_face_identity"
    description = "Face identity similarity using DINOv2 on face crops (better than ArcFace for AI-generated)"
    default_config = {
        "model_name": "dinov2_vitb14",
        "face_model": "buffalo_l",
        "subsample": 8,
        "face_margin": 0.3,
        "warning_threshold": 0.3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "dinov2_vitb14")
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.subsample = self.config.get("subsample", 8)
        self.face_margin = self.config.get("face_margin", 0.3)
        self.warning_threshold = self.config.get("warning_threshold", 0.3)
        self._dino = None
        self._transform = None
        self._face_app = None
        self._device = "cpu"

    def setup(self):
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load DINOv2
        try:
            self._dino = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self._dino.eval().to(self._device)
            logger.info(f"DINOFaceIdentity: loaded {self.model_name} on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")
            return

        # Transform
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Face detector
        try:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(name=self.face_model, providers=["CPUExecutionProvider"])
            self._face_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("DINOFaceIdentity: face detector ready")
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")

    def process(self, sample: Sample) -> Sample:
        if self._dino is None or self._face_app is None:
            return sample

        # Need reference
        if sample.reference_path is None:
            return sample

        try:
            ref_embedding = self._compute_reference_embedding(sample.reference_path)
            if ref_embedding is None:
                return sample

            frames = self._load_frames(sample)
            if not frames:
                return sample

            similarities = []
            for frame in frames:
                emb = self._extract_face_dino(frame)
                if emb is not None:
                    sim = float(np.dot(emb, ref_embedding))
                    similarities.append(sim)

            if not similarities:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="No faces detected in any frame",
                        details={"frames_checked": len(frames)},
                    )
                )
                return sample

            mean_sim = float(np.mean(similarities))
            max_sim = float(np.max(similarities))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dino_face_identity = mean_sim
            sample.quality_metrics.dino_face_identity_max = max_sim

            if mean_sim < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low DINO face identity (score={mean_sim:.3f})",
                        details={
                            "dino_face_identity": mean_sim,
                            "dino_face_identity_max": max_sim,
                            "faces_found": len(similarities),
                            "frames_checked": len(frames),
                        },
                        recommendation="Identity may not be well preserved. Check if the correct person appears.",
                    )
                )

        except Exception as e:
            logger.warning(f"DINOFaceIdentity failed for {sample.path}: {e}")

        return sample

    def _compute_reference_embedding(self, reference_path) -> Optional[np.ndarray]:
        """Compute DINO face embedding from reference image."""
        from pathlib import Path
        ref_path = Path(reference_path)

        if ref_path.is_dir():
            # Average over all images in directory
            embeddings = []
            for img_path in sorted(ref_path.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                emb = self._extract_face_dino(img)
                if emb is not None:
                    embeddings.append(emb)
            if not embeddings:
                return None
            avg = np.mean(embeddings, axis=0)
            return avg / np.linalg.norm(avg)
        else:
            # Single image
            img = cv2.imread(str(ref_path))
            if img is None:
                return None
            return self._extract_face_dino(img)

    def _extract_face_dino(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face, crop, encode with DINOv2."""
        import torch

        faces = self._face_app.get(frame)
        if not faces:
            return None

        # Take largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(c) for c in largest.bbox]

        # Add margin
        margin = int((x2 - x1) * self.face_margin)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        tensor = self._transform(crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb = self._dino(tensor).cpu().numpy()[0]

        return emb / np.linalg.norm(emb)

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        """Load frames from video or image."""
        from pathlib import Path
        path = Path(sample.path)

        if sample.is_video:
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, self.subsample, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            return frames
        else:
            img = cv2.imread(str(path))
            return [img] if img is not None else []

    def on_dispose(self) -> None:
        """Release GPU memory."""
        self._dino = None
        self._face_app = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
