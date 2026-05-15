"""EntityBench — cross-shot identity persistence (arXiv:2605.15199).

Whereas existing ``subject_consistency`` measures within-video frame
similarity, EntityBench scores whether the *same entity* persists across
*multiple shots/samples*. Samples are grouped by ``reference_path``'s
parent directory (or, if absent, by their own parent directory) — each
group is treated as one multi-shot sequence describing the same entity.

Outputs (per sample, identical within a group):
    * ``entitybench_identity_consistency``  — face/identity persistence.
    * ``entitybench_appearance_consistency`` — overall visual persistence.

Tiered backend:
    1. DINOv2 visual + ArcFace/MagFace face embeddings, clustered per group.
    2. CLIP visual embedding cosine across the group.
    3. Color-histogram correlation heuristic.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class EntityBenchModule(BatchMetricModule):
    name = "entitybench"
    description = "EntityBench cross-shot identity persistence (arXiv:2605.15199)"
    default_config = {
        "backend": "auto",  # "auto" | "dinov2_face" | "clip" | "histogram"
        "models_dir": "models",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = self.config.get("backend", "auto")
        self.models_dir = self.config.get("models_dir", "models")
        self._device = "cpu"
        self._dinov2 = None
        self._dinov2_proc = None
        self._face_model = None
        self._clip_model = None
        self._clip_processor = None
        self.active_backend = "histogram"
        # Group key → list of (sample, embedding) tuples
        self._groups: Dict[str, List[Tuple[Sample, Optional[np.ndarray], Optional[np.ndarray]]]] = defaultdict(list)

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return
        if self.backend_pref in ("auto", "dinov2_face") and self._try_setup_dinov2_face():
            self.active_backend = "dinov2_face"
        elif self.backend_pref in ("auto", "clip") and self._try_setup_clip():
            self.active_backend = "clip"
        else:
            self.active_backend = "histogram"
        logger.info(f"EntityBench initialized with backend={self.active_backend}")

    def _try_setup_dinov2_face(self) -> bool:
        try:
            from transformers import AutoModel, AutoImageProcessor
            self._dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self._dinov2 = AutoModel.from_pretrained("facebook/dinov2-base").to(self._device).eval()
        except Exception as e:
            logger.debug(f"EntityBench: DINOv2 unavailable: {e}")
            return False
        # Face embeddings are optional — DINOv2 alone is still a valid tier.
        try:
            import insightface
            self._face_model = insightface.app.FaceAnalysis(name="buffalo_l")
            self._face_model.prepare(ctx_id=0 if self._device == "cuda" else -1)
        except Exception as e:
            logger.debug(f"EntityBench: InsightFace unavailable: {e}")
            self._face_model = None
        return True

    def _try_setup_clip(self) -> bool:
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=self.models_dir,
            ).to(self._device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=self.models_dir,
            )
            return True
        except Exception as e:
            logger.debug(f"EntityBench: CLIP unavailable: {e}")
            return False

    # BatchMetricModule contract -----------------------------------------
    def extract_features(self, sample: Sample) -> Optional[object]:
        # Compute per-sample (visual, face) embedding so we can cluster
        # them per group in on_dispose. Returns a tuple so the base class's
        # cache works, though we maintain a richer group structure ourselves.
        img = sample.load_image()
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visual = self._visual_embed(img_rgb)
        face = self._face_embed(img_rgb) if self._face_model is not None else None
        group_key = _group_key(sample)
        self._groups[group_key].append((sample, visual, face))
        return visual if visual is not None else np.zeros((1,))

    def compute_distribution_metric(self, features, reference_features=None) -> float:
        # Implemented for the abstract contract; per-group aggregation
        # happens in on_dispose so individual samples can be written.
        return 0.0

    def on_dispose(self) -> None:
        if not self._groups:
            return
        for group, items in self._groups.items():
            if len(items) < 2:
                # Single-shot group → identity is trivially preserved.
                self._write_group(items, 1.0, 1.0)
                continue
            id_score = self._identity_consistency(items)
            app_score = self._appearance_consistency(items)
            self._write_group(items, id_score, app_score)
        self._groups.clear()

    # --------------------------------------------------------------------
    def _write_group(self, items, id_score: float, app_score: float) -> None:
        for sample, _, _ in items:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.entitybench_identity_consistency = round(float(id_score), 3)
            sample.quality_metrics.entitybench_appearance_consistency = round(float(app_score), 3)

    def _identity_consistency(self, items) -> float:
        faces = [face for _, _, face in items if face is not None]
        if len(faces) >= 2:
            return _pairwise_mean_cosine(faces)
        # No face embeddings → fall back to visual cosine
        visuals = [v for _, v, _ in items if v is not None]
        if len(visuals) >= 2:
            return _pairwise_mean_cosine(visuals)
        return self._histogram_consistency(items)

    def _appearance_consistency(self, items) -> float:
        visuals = [v for _, v, _ in items if v is not None]
        if len(visuals) >= 2:
            return _pairwise_mean_cosine(visuals)
        return self._histogram_consistency(items)

    def _histogram_consistency(self, items) -> float:
        hists = []
        for sample, _, _ in items:
            img = sample.load_image()
            if img is None:
                continue
            h = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            cv2.normalize(h, h)
            hists.append(h.flatten())
        if len(hists) < 2:
            return 0.5
        sims = []
        for i in range(len(hists)):
            for j in range(i + 1, len(hists)):
                # OpenCV correlation in [-1, 1] → map to [0, 1]
                sim = float(np.dot(hists[i], hists[j]) /
                            (np.linalg.norm(hists[i]) * np.linalg.norm(hists[j]) + 1e-9))
                sims.append((sim + 1.0) / 2.0)
        return float(np.mean(sims))

    def _visual_embed(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.active_backend == "dinov2_face" and self._dinov2 is not None:
            try:
                import torch
                inputs = self._dinov2_proc(images=img_rgb, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out = self._dinov2(**inputs)
                return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            except Exception as e:
                logger.debug(f"DINOv2 embed failed: {e}")
        if self.active_backend == "clip" and self._clip_model is not None:
            try:
                import torch
                from PIL import Image as PILImage
                inputs = self._clip_processor(
                    images=PILImage.fromarray(img_rgb), return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    feat = self._clip_model.get_image_features(**inputs)
                return feat.cpu().numpy().flatten()
            except Exception as e:
                logger.debug(f"CLIP embed failed: {e}")
        return None

    def _face_embed(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._face_model is None:
            return None
        try:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            faces = self._face_model.get(img_bgr)
            if not faces:
                return None
            # Use the most confident face
            faces.sort(key=lambda f: -float(getattr(f, "det_score", 0.0)))
            return faces[0].normed_embedding
        except Exception as e:
            logger.debug(f"Face embed failed: {e}")
            return None


def _group_key(sample: Sample) -> str:
    # Prefer the reference_path's parent (caller groups by shared reference);
    # otherwise group by the sample's own parent directory.
    ref = getattr(sample, "reference_path", None)
    if ref is not None:
        return str(Path(ref).parent)
    return str(Path(sample.path).parent)


def _pairwise_mean_cosine(embs: List[np.ndarray]) -> float:
    if len(embs) < 2:
        return 1.0
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            a, b = embs[i], embs[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            sims.append(float(np.dot(a, b) / denom))
    # Map cosine [-1, 1] → [0, 1]
    return float(np.clip((np.mean(sims) + 1.0) / 2.0, 0.0, 1.0))
