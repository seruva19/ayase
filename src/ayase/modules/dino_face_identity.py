"""DINO Face Identity — reference-based identity indicator using DINOv2 face crops.

DINOv2 on face crops captures overall facial appearance and complements
face-recognition embeddings. Note: it is a weaker identity *discriminator* than
ArcFace — face-recognition embeddings separate genuine from impostor identities
more reliably (higher true-accept rate at a fixed false-accept budget), so prefer
ArcFace (see the ``identity_loss`` / ``face_recognition_score`` module) as the
identity gate and use this metric as a complementary appearance indicator.
Applicable to identity-preserving generation (LoRA, DreamBooth, IP-Adapter,
InstantID).

Outputs:
    dino_face_identity     — cosine similarity 0-1 (higher = better match)
    dino_face_identity_max — max similarity across sampled frames

Requires ``sample.reference_path`` pointing to a reference face image or directory.
Gracefully skips when no reference is provided.

Backends:
    - Face detection: InsightFace (buffalo_l)
    - Embedding: DINOv2 ViT-B/14 (facebookresearch/dinov2 architecture; weights from
      the ayase-models HF mirror)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# DINOv2 backbone weights are fetched from the ayase-models HF mirror rather than
# the torch.hub entrypoint's fbaipublicfiles original, which is unreliable on some
# networks (and bypasses the mirror the rest of the pipeline already relies on).
# The architecture still comes from the (reliable, cacheable) torch.hub repo code.
_DINOV2_MIRROR_BASE = "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
# Only sizes actually rehosted on the mirror; any other ``model_name`` falls back to
# the torch.hub pretrained path (upstream fbaipublicfiles).
_DINOV2_WEIGHTS = {
    "dinov2_vitb14": "dino_face_identity/dinov2_vitb14_pretrain.pth",
}


class DINOFaceIdentityModule(PipelineModule):
    name = "dino_face_identity"
    description = "Face identity similarity via DINOv2 on face crops (appearance indicator; ArcFace is the stronger identity discriminator)"
    default_config = {
        "model_name": "dinov2_vitb14",
        "face_model": "buffalo_l",
        "subsample": 8,
        "face_margin": 0.3,
        "warning_threshold": 0.3,
        "models_dir": "models",
    }
    metric_groups = {
        "dino_face_identity": "face",
        "dino_face_identity_max": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "dinov2_vitb14")
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.subsample = self.config.get("subsample", 8)
        self.face_margin = self.config.get("face_margin", 0.3)
        self.warning_threshold = self.config.get("warning_threshold", 0.3)
        self.models_dir = self.config.get("models_dir", "models")
        self._dino = None
        self._transform = None
        self._face_app = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self):
        import torch
        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))

        # Load DINOv2. The architecture comes from the torch.hub repo code; the
        # weights come from the ayase-models HF mirror (reliable, and consistent with
        # the rest of the pipeline) instead of the torch.hub fbaipublicfiles original.
        try:
            rel = _DINOV2_WEIGHTS.get(self.model_name)
            if rel:
                from ayase.config import download_model_file

                self._dino = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, pretrained=False
                )
                ckpt = download_model_file(rel, _DINOV2_MIRROR_BASE + rel, self.models_dir)
                self._dino.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            else:
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
            return

        # Both DINOv2 embedder and the face detector are ready.
        self._backend = f"dinov2:{self.model_name}+insightface"

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
        """Uniformly sampled BGR frames (image → one) from the shared cache.

        InsightFace and the crop path only read these arrays, so the read-only
        cache views are safe to pass through.
        """
        return sample_frames(sample.path, max_frames=self.subsample, color="bgr")

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
