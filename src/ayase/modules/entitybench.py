"""EntityBench — cross-shot identity persistence (arXiv:2605.15199).

Whereas existing ``subject_consistency`` measures within-video frame
similarity, EntityBench scores whether the *same entity* persists across
*multiple shots/samples*. Samples are grouped by ``reference_path``'s
parent directory (or, if absent, by their own parent directory) — each
group is treated as one multi-shot sequence describing the same entity.

Outputs (per sample, identical within a group):
    * ``entitybench_identity_consistency``  — face/identity persistence.
    * ``entitybench_appearance_consistency`` — overall visual persistence.

Backends (real visual/identity embeddings only):
    1. DINOv2 visual + ArcFace/MagFace face embeddings, clustered per group.
    2. CLIP visual embedding cosine across the group.

When neither DINOv2 nor CLIP is available the metric is left unset — a
color-histogram correlation is NOT used as a stand-in for identity/appearance
consistency.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import arrays_to_pil
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import cached_clip_image_features, media_state_key

logger = logging.getLogger(__name__)


class EntityBenchModule(BatchMetricModule):
    name = "entitybench"
    description = "EntityBench cross-shot identity persistence (arXiv:2605.15199)"
    default_config = {
        "backend": "auto",  # "auto" | "dinov2_face" | "clip"
        "clip_model": "openai/clip-vit-base-patch32",
        "models_dir": "models",
    }
    metric_groups = {
        "entitybench_appearance_consistency": "temporal",
        "entitybench_identity_consistency": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend_pref = self.config.get("backend", "auto")
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.models_dir = self.config.get("models_dir", "models")
        self._device = "cpu"
        self._dinov2 = None
        self._dinov2_proc = None
        self._face_model = None
        self._clip_model = None
        self._clip_processor = None
        self.active_backend = "unavailable"
        # Group key → list of (sample, embedding) tuples
        self._groups: Dict[str, List[Tuple[Sample, Optional[np.ndarray], Optional[np.ndarray]]]] = defaultdict(list)

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except ImportError:
            self.active_backend = "unavailable"
            return
        if self.backend_pref in ("auto", "dinov2_face") and self._try_setup_dinov2_face():
            self.active_backend = "dinov2_face"
        elif self.backend_pref in ("auto", "clip") and self._try_setup_clip():
            self.active_backend = "clip"
        else:
            self.active_backend = "unavailable"
            logger.warning(
                "EntityBench: no real embedding backend available "
                "(install transformers for DINOv2/CLIP) — metric left unset."
            )
        logger.info(f"EntityBench initialized with backend={self.active_backend}")

    def _try_setup_dinov2_face(self) -> bool:
        try:
            from transformers import AutoModel, AutoImageProcessor
            from ayase.runtime import from_pretrained_with_attention, shared_runtime_resource

            model_name = "facebook/dinov2-base"

            def load_dino():
                proc = AutoImageProcessor.from_pretrained(model_name)
                model = from_pretrained_with_attention(
                    AutoModel,
                    model_name,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
                ).to(self._device).eval()
                return model, proc

            # Share the DINOv2-base backbone with other modules (e.g.
            # subject_consistency) via the pipeline runtime resource cache.
            self._dinov2, self._dinov2_proc = shared_runtime_resource(
                self,
                (
                    "hf_vision",
                    model_name,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "safetensors",
                ),
                load_dino,
            )
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
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            resolved = resolve_model_path(self.clip_model_name, self.models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
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
        if self.active_backend == "unavailable":
            return None
        img = sample.load_image()
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visual = self._visual_embed(sample, img_rgb)
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
    def _write_group(self, items, id_score: Optional[float], app_score: Optional[float]) -> None:
        for sample, _, _ in items:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            if id_score is not None:
                sample.quality_metrics.entitybench_identity_consistency = round(float(id_score), 3)
            if app_score is not None:
                sample.quality_metrics.entitybench_appearance_consistency = round(float(app_score), 3)

    def _identity_consistency(self, items) -> Optional[float]:
        faces = [face for _, _, face in items if face is not None]
        if len(faces) >= 2:
            return _pairwise_mean_cosine(faces)
        # No face embeddings → fall back to real visual (DINOv2/CLIP) cosine.
        visuals = [v for _, v, _ in items if v is not None]
        if len(visuals) >= 2:
            return _pairwise_mean_cosine(visuals)
        # No real embeddings available → do not fabricate a score.
        return None

    def _appearance_consistency(self, items) -> Optional[float]:
        visuals = [v for _, v, _ in items if v is not None]
        if len(visuals) >= 2:
            return _pairwise_mean_cosine(visuals)
        return None

    def _visual_embed(self, sample: Sample, img_rgb: np.ndarray) -> Optional[np.ndarray]:
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
                feat = cached_clip_image_features(
                    self,
                    self._clip_model,
                    self._clip_processor,
                    arrays_to_pil([img_rgb]),
                    model_key=self.clip_model_name,
                    device=self._device,
                    cache_key=("entitybench", media_state_key(sample.path)),
                )
                return feat.squeeze(0).detach().float().cpu().numpy().flatten()
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
