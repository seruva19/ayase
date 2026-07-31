"""AdaFace identity similarity — reference-based face identity preservation (CVPR 2022).

Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition". AdaFace
weights the margin by an image-quality proxy (the feature norm) during training,
which makes its embeddings noticeably more reliable than ArcFace on low-quality,
small or blurred faces.

This module is the second, independent identity backbone next to the ArcFace-based
``identity_loss`` / ``face_recognition_score``: identity-preserving generators
(InstantID, PuLID, IP-Adapter-FaceID) are conditioned on ArcFace embeddings, so an
ArcFace-only score partly measures the encoder the generator was trained against.
AdaFace comes from a different training objective and is reported alongside ArcFace
in that literature for exactly this reason.

Output:
    adaface_identity_similarity — cosine similarity to the reference face,
                                  clipped to 0-1 (higher = better preservation)

Requires ``sample.reference_path`` (a face image or a directory of face images);
gracefully skips when no reference is provided.

Backend (real face-recognition embeddings only, no heuristic fallback):
    - Detection + 5-point alignment: InsightFace (buffalo_l) ``norm_crop`` to the
      standard 112x112 ArcFace template — the same template AdaFace's own MTCNN
      alignment targets.
    - Embedding: official AdaFace IResNet backbone (vendored from CVLface, MIT)
      with author-published weights pinned by HuggingFace revision.

Pre-cropped face chips (a 112x112 aligned face filling the frame) are invisible to
RetinaFace, so detection is retried once on a replicate-padded copy (``pad_retry``).

Without InsightFace or the weights the metric is left ``None``.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.faces import detect_largest_face
from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Author-published AdaFace checkpoints (mk-minchul), pinned by commit so a
# re-run downloads byte-identical weights. Keys are ``<backbone>_<train set>``.
_ADAFACE_CHECKPOINTS = {
    "ir101_webface12m": {
        "repo": "minchul/cvlface_adaface_ir101_webface12m",
        "revision": "54f602a0737bd1ee4a4e7e9fd089a485f397fefd",
        "arch": "ir101",
        "size": "261.0 MB",
    },
    "ir101_webface4m": {
        "repo": "minchul/cvlface_adaface_ir101_webface4m",
        "revision": "f2b38d9e24bfe301490d8dd081d8924b102333dd",
        "arch": "ir101",
        "size": "261.0 MB",
    },
    "ir101_ms1mv2": {
        "repo": "minchul/cvlface_adaface_ir101_ms1mv2",
        "revision": "afdb94f8190f4cd8ea1467258ce65f1d76033b63",
        "arch": "ir101",
        "size": "261.0 MB",
    },
    "ir50_webface4m": {
        "repo": "minchul/cvlface_adaface_ir50_webface4m",
        "revision": "60a65befbcf7e19284c4f3ac730f56867ed29594",
        "arch": "ir50",
        "size": "175.4 MB",
    },
    "ir18_webface4m": {
        "repo": "minchul/cvlface_adaface_ir18_webface4m",
        "revision": "0dd53f188fa27968b0a1326970ebf4aeb37ce2ca",
        "arch": "ir18",
        "size": "97.1 MB",
    },
}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _weight_url(entry: dict) -> str:
    return (
        f"https://huggingface.co/{entry['repo']}/resolve/"
        f"{entry['revision']}/model.safetensors"
    )


class AdaFaceModule(PipelineModule):
    name = "adaface"
    description = "AdaFace identity similarity vs reference face (CVPR 2022, quality-adaptive margin)"
    default_config = {
        "checkpoint": "ir101_webface12m",
        "face_model": "buffalo_l",
        "subsample": 8,
        "warning_threshold": 0.3,
        # Retry detection on a replicate-padded copy when a tightly cropped face
        # leaves the detector no context (0 disables the retry).
        "pad_retry": 0.25,
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": entry["repo"],
            "type": "huggingface",
            "url": _weight_url(entry),
            "task": f"AdaFace {key} face-recognition embedding",
            "size": entry["size"],
            "auto_download": True,
            "notes": f"MIT (CVLface); pinned to revision {entry['revision']}",
        }
        for key, entry in _ADAFACE_CHECKPOINTS.items()
    ]
    metric_info = {
        "adaface_identity_similarity": (
            "AdaFace cosine similarity to the reference face (0-1, higher=better)"
        ),
    }
    metric_groups = {"adaface_identity_similarity": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self.checkpoint = str(self.config.get("checkpoint", "ir101_webface12m"))
        self.face_model = str(self.config.get("face_model", "buffalo_l"))
        self.subsample = int(self.config.get("subsample", 8))
        self.warning_threshold = float(self.config.get("warning_threshold", 0.3))
        self.pad_retry = max(0.0, float(self.config.get("pad_retry", 0.25)))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._net = None
        self._face_app = None
        self._norm_crop = None
        self._torch = None
        self._device = "cpu"
        self._backend = "unavailable"

    # -- Setup ------------------------------------------------------------------

    def setup(self) -> None:
        if self.test_mode:
            logger.debug("AdaFace: test mode, skipping model setup")
            return

        entry = _ADAFACE_CHECKPOINTS.get(self.checkpoint)
        if entry is None:
            logger.warning(
                "AdaFace: unknown checkpoint %r (known: %s)",
                self.checkpoint,
                ", ".join(sorted(_ADAFACE_CHECKPOINTS)),
            )
            return

        try:
            import torch
            from safetensors.torch import load_file

            from ayase.config import download_model_file
            from ayase.runtime import resolve_torch_device
            from ayase.third_party.adaface import IR_18, IR_50, IR_101

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            weights = download_model_file(
                f"adaface/{self.checkpoint}/model.safetensors",
                _weight_url(entry),
                self.models_dir,
            )
            builder = {"ir18": IR_18, "ir50": IR_50, "ir101": IR_101}[entry["arch"]]
            net = builder(input_size=(112, 112), output_dim=512)

            # CVLface stores the wrapper state dict; the backbone lives under
            # ``model.net.``. Loading is strict so a checkpoint/architecture
            # mismatch fails loudly instead of silently scoring random weights.
            state = load_file(str(weights))
            prefix = "model.net."
            backbone_state = {
                k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
            }
            if not backbone_state:
                raise RuntimeError(f"no '{prefix}*' tensors in {weights}")
            net.load_state_dict(backbone_state, strict=True)
            net.eval().to(self._device)
            self._net = net
            self._torch = torch
        except ImportError as exc:
            logger.warning("AdaFace requires torch and safetensors: %s", exc)
            return
        except Exception as exc:
            logger.warning("AdaFace backbone initialisation failed: %s", exc)
            return

        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align

            self._norm_crop = face_align.norm_crop
            self._face_app = FaceAnalysis(
                name=self.face_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(
                ctx_id=0 if self._device.startswith("cuda") else -1, det_size=(640, 640)
            )
        except Exception as exc:
            self._net = None
            logger.warning(
                "AdaFace needs InsightFace for detection/alignment "
                "(pip install insightface onnxruntime): %s",
                exc,
            )
            return

        self._backend = f"adaface:{self.checkpoint}+insightface"
        logger.info(
            "AdaFace initialised (%s) on %s", self.checkpoint, self._device
        )

    # -- Processing -------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if self._net is None or self._face_app is None:
            return sample
        if sample.reference_path is None:
            return sample

        try:
            ref_embedding = self._reference_embedding(sample.reference_path)
            if ref_embedding is None:
                logger.debug("AdaFace: no face found in reference %s", sample.reference_path)
                return sample

            frames = self._load_frames(sample)
            if not frames:
                return sample

            similarities = []
            for frame in frames:
                emb = self._embed_face(frame)
                if emb is not None:
                    similarities.append(float(np.dot(ref_embedding, emb)))

            if not similarities:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="AdaFace: no face detected in any frame",
                        details={"frames_checked": len(frames)},
                    )
                )
                return sample

            score = float(np.clip(np.mean(similarities), 0.0, 1.0))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.adaface_identity_similarity = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low AdaFace identity similarity ({score:.3f})",
                        details={
                            "adaface_identity_similarity": score,
                            "faces_found": len(similarities),
                            "frames_checked": len(frames),
                        },
                        recommendation="Identity may not be preserved — verify the reference face matches.",
                    )
                )
        except Exception as exc:
            logger.warning("AdaFace failed for %s: %s", sample.path, exc)

        return sample

    # -- Embeddings -------------------------------------------------------------

    def _reference_embedding(self, reference_path) -> Optional[np.ndarray]:
        """Unit-norm AdaFace embedding of the reference (image or directory)."""
        ref_path = Path(reference_path)

        if ref_path.is_dir():
            embeddings = []
            for img_path in sorted(ref_path.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                emb = self._embed_face(img)
                if emb is not None:
                    embeddings.append(emb)
            if not embeddings:
                return None
            avg = np.mean(embeddings, axis=0)
            return avg / (np.linalg.norm(avg) + 1e-10)

        img = cv2.imread(str(ref_path))
        if img is None:
            return None
        return self._embed_face(img)

    def _detect_largest_face(self, frame_bgr: np.ndarray):
        """Largest face plus the image its keypoints refer to (see ``ayase.faces``)."""
        return detect_largest_face(self._face_app, frame_bgr, self.pad_retry)

    def _embed_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest face, align to 112x112, return a unit-norm embedding."""
        face, detect_image = self._detect_largest_face(frame_bgr)
        if face is None:
            return None
        kps = getattr(face, "kps", None)
        if kps is None:
            return None

        # ArcFace 5-point alignment → fresh 112x112 BGR crop.
        aligned_bgr = self._norm_crop(detect_image, landmark=kps, image_size=112)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        torch = self._torch
        tensor = torch.from_numpy(aligned_rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5  # AdaFace: mean=0.5, std=0.5, RGB, 112x112
        tensor = tensor.to(self._device)

        with torch.no_grad():
            emb = self._net(tensor).float().cpu().numpy()[0]

        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return emb / norm

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        """Uniformly sampled BGR frames (a single frame for images)."""
        return sample_frames(sample.path, max_frames=self.subsample, color="bgr")

    def on_dispose(self) -> None:
        self._net = None
        self._face_app = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
