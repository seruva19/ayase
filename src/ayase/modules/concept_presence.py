"""Concept Presence Detection — face/object/style detection in images and video.

Detects the presence of specified concepts in images via face detection and/or
CLIP-based semantic matching.  In ``"auto"`` mode the module selects face
detection when concept keywords contain face-related terms, and CLIP-based
matching otherwise.

Outputs:
    concept_presence    — max confidence across checked concepts (0-1)
    concept_count       — number of detected concept instances
    concept_face_count  — number of faces detected

Tiered backends (face):
    1. **InsightFace** — industry-standard face detector/embedder
    2. **MediaPipe** FaceDetection — lightweight fallback
    3. **OpenCV Haar cascade** — always available

Tiered backends (CLIP):
    1. **transformers** CLIPModel — HuggingFace CLIP
    2. **Heuristic** — color-histogram template matching proxy
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Keywords that trigger face-detection mode in "auto"
_FACE_KEYWORDS = frozenset([
    "face", "faces", "person", "people", "man", "woman", "child", "human",
    "portrait", "selfie", "headshot", "smile", "expression",
])


class ConceptPresenceModule(PipelineModule):
    name = "concept_presence"
    description = "Detect concept presence via face detection, CLIP-based object/style detection"
    default_config = {
        "detection_mode": "auto",  # "auto", "face", "clip", "combined"
        "clip_model": "openai/clip-vit-base-patch32",
        "clip_threshold": 0.25,  # CLIP similarity threshold for concept match
        "face_detection_confidence": 0.5,  # Face detection confidence threshold
        "concepts": [],  # Concepts to check (empty = derive from caption)
        "num_frames": 5,  # Frames to sample from video
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False

        # Face detection backend state
        self._face_backend: Optional[str] = None  # "insightface" | "mediapipe" | "haar"
        self._face_app = None  # InsightFace FaceAnalysis
        self._mp_face_det = None  # MediaPipe FaceDetection
        self._haar_cascade = None  # OpenCV Haar cascade

        # CLIP backend state
        self._clip_backend: Optional[str] = None  # "transformers" | "heuristic"
        self._clip_model = None
        self._clip_processor = None
        self._clip_device = "cpu"

    def setup(self) -> None:
        """Initialize face detection and CLIP backends with tiered fallback."""
        self._setup_face_backend()
        self._setup_clip_backend()

        # Module is available if at least one backend is ready
        if self._face_backend is not None or self._clip_backend is not None:
            self._ml_available = True

    def _setup_face_backend(self) -> None:
        """Try face detection backends in order of preference."""
        # Tier 1: InsightFace
        try:
            from insightface.app import FaceAnalysis

            self._face_app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            self._face_app.prepare(ctx_id=-1, det_size=(640, 640))
            self._face_backend = "insightface"
            logger.info("ConceptPresence: face backend = InsightFace")
            return
        except Exception:
            pass

        # Tier 2: MediaPipe FaceDetection
        try:
            import mediapipe as mp

            self._mp_face_det = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=self.config.get("face_detection_confidence", 0.5)
            )
            self._face_backend = "mediapipe"
            logger.info("ConceptPresence: face backend = MediaPipe")
            return
        except Exception:
            pass

        # Tier 3: OpenCV Haar cascade (always available via cv2)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if not self._haar_cascade.empty():
                self._face_backend = "haar"
                logger.info("ConceptPresence: face backend = OpenCV Haar")
                return
        except Exception:
            pass

        logger.info("ConceptPresence: no face backend available")

    def _setup_clip_backend(self) -> None:
        """Try CLIP backends in order of preference."""
        # Tier 1: transformers CLIP
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")

            # Try to resolve local model path
            try:
                from ayase.config import resolve_model_path

                models_dir = self.config.get("models_dir", "models")
                resolved = resolve_model_path(clip_model_name, models_dir)
            except Exception:
                resolved = clip_model_name

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = CLIPModel.from_pretrained(resolved).to(device)
            self._clip_processor = CLIPProcessor.from_pretrained(resolved)
            self._clip_device = device
            self._clip_backend = "transformers"
            logger.info("ConceptPresence: CLIP backend = transformers on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("Transformers CLIP unavailable: %s", e)

        # Tier 2: Heuristic (color-histogram matching)
        self._clip_backend = "heuristic"
        logger.info("ConceptPresence: using heuristic fallback for concept matching")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        if not self._ml_available:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            # Determine concepts to check
            concepts = list(self.config.get("concepts", []))
            if not concepts and sample.caption:
                # Extract concepts from caption text
                concepts = [sample.caption.text]

            # Determine detection mode
            mode = self.config.get("detection_mode", "auto")
            if mode == "auto":
                mode = self._auto_detect_mode(concepts)

            face_count = 0
            concept_confidence = 0.0
            concept_count = 0

            # Face detection
            if mode in ("face", "combined"):
                face_count = self._detect_faces(frames)
                if face_count > 0:
                    concept_confidence = max(concept_confidence, 1.0)
                    concept_count += face_count

            # CLIP-based concept detection
            if mode in ("clip", "combined") and concepts:
                clip_conf, clip_count = self._detect_concepts_clip(frames, concepts)
                concept_confidence = max(concept_confidence, clip_conf)
                concept_count += clip_count

            sample.quality_metrics.concept_presence = float(
                np.clip(concept_confidence, 0.0, 1.0)
            )
            sample.quality_metrics.concept_count = concept_count
            sample.quality_metrics.concept_face_count = face_count

        except Exception as e:
            logger.warning("ConceptPresence failed for %s: %s", sample.path, e)

        return sample

    # -- Mode detection ---------------------------------------------------------

    def _auto_detect_mode(self, concepts: List[str]) -> str:
        """Determine detection mode based on concept keywords."""
        if not concepts:
            # No concepts specified; default to face if face backend available
            if self._face_backend is not None:
                return "face"
            return "clip"

        # Check if any concept contains face-related keywords
        for concept in concepts:
            words = set(concept.lower().split())
            if words & _FACE_KEYWORDS:
                if self._face_backend is not None and self._clip_backend is not None:
                    return "combined"
                if self._face_backend is not None:
                    return "face"

        return "clip"

    # -- Face detection ---------------------------------------------------------

    def _detect_faces(self, frames: List[np.ndarray]) -> int:
        """Detect faces across frames and return total face count."""
        if self._face_backend is None:
            return 0

        counts = []
        for frame in frames:
            if self._face_backend == "insightface":
                count = self._detect_faces_insightface(frame)
            elif self._face_backend == "mediapipe":
                count = self._detect_faces_mediapipe(frame)
            else:
                count = self._detect_faces_haar(frame)
            counts.append(count)

        # Return max face count across frames
        return max(counts) if counts else 0

    def _detect_faces_insightface(self, frame: np.ndarray) -> int:
        try:
            faces = self._face_app.get(frame)
            return len(faces)
        except Exception:
            return 0

    def _detect_faces_mediapipe(self, frame: np.ndarray) -> int:
        try:
            results = self._mp_face_det.process(frame)
            if results.detections:
                return len(results.detections)
            return 0
        except Exception:
            return 0

    def _detect_faces_haar(self, frame: np.ndarray) -> int:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            confidence = self.config.get("face_detection_confidence", 0.5)
            # Adjust minNeighbors based on confidence (higher confidence = more neighbors)
            min_neighbors = max(3, int(confidence * 10))
            faces = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(30, 30)
            )
            return len(faces)
        except Exception:
            return 0

    # -- CLIP concept detection -------------------------------------------------

    def _detect_concepts_clip(
        self, frames: List[np.ndarray], concepts: List[str]
    ) -> tuple:
        """Detect concepts using CLIP or heuristic.

        Returns (max_confidence, count_above_threshold).
        """
        if self._clip_backend == "transformers":
            return self._detect_concepts_transformers(frames, concepts)
        return self._detect_concepts_heuristic(frames, concepts)

    def _detect_concepts_transformers(
        self, frames: List[np.ndarray], concepts: List[str]
    ) -> tuple:
        """Detect concepts using HuggingFace CLIP."""
        try:
            import torch

            threshold = self.config.get("clip_threshold", 0.25)

            # Encode text concepts
            text_inputs = self._clip_processor(
                text=concepts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._clip_device)

            with torch.no_grad():
                text_features = self._clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            # Compute similarity for each frame
            max_confidence = 0.0
            concept_detections = set()

            for frame in frames:
                pil_image = Image.fromarray(frame)
                image_inputs = self._clip_processor(
                    images=pil_image,
                    return_tensors="pt",
                ).to(self._clip_device)

                with torch.no_grad():
                    image_features = self._clip_model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(
                        p=2, dim=-1, keepdim=True
                    )
                    similarities = (image_features @ text_features.T).squeeze(0)

                for idx, sim in enumerate(similarities):
                    sim_val = sim.item()
                    if sim_val > max_confidence:
                        max_confidence = sim_val
                    if sim_val >= threshold:
                        concept_detections.add(idx)

            return max_confidence, len(concept_detections)

        except Exception as e:
            logger.debug("CLIP concept detection failed: %s", e)
            return 0.0, 0

    def _detect_concepts_heuristic(
        self, frames: List[np.ndarray], concepts: List[str]
    ) -> tuple:
        """Heuristic concept presence via color-histogram complexity.

        Returns a rough proxy: high complexity = likely has some concept present.
        Cannot distinguish between specific concepts.
        """
        try:
            threshold = self.config.get("clip_threshold", 0.25)
            confidences = []

            for frame in frames:
                # Compute color histogram complexity as a proxy for content richness
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])

                # Normalize
                hist_h = hist_h / (hist_h.sum() + 1e-10)
                hist_s = hist_s / (hist_s.sum() + 1e-10)

                # Entropy as content richness proxy
                h_entropy = float(-np.sum(hist_h[hist_h > 0] * np.log2(hist_h[hist_h > 0])))
                s_entropy = float(-np.sum(hist_s[hist_s > 0] * np.log2(hist_s[hist_s > 0])))

                # Normalize to 0-1 range
                # Max entropy for 180 bins ~= 7.5, for 256 bins ~= 8.0
                confidence = min(1.0, (h_entropy / 7.5 + s_entropy / 8.0) / 2.0)
                confidences.append(confidence)

            max_conf = max(confidences) if confidences else 0.0
            count = sum(1 for c in confidences if c >= threshold)
            # Heuristic can only say "something is there", count = 1 or 0
            return max_conf, min(count, len(concepts))

        except Exception as e:
            logger.debug("Heuristic concept detection failed: %s", e)
            return 0.0, 0

    # -- Frame loading ----------------------------------------------------------

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        """Load RGB frames from image or video."""
        frames = []
        try:
            if not sample.is_video:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                num_frames = self.config.get("num_frames", 5)
                n = min(num_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
        except Exception as e:
            logger.debug("Frame loading failed: %s", e)
        return frames
