"""CSIM: cosine similarity of face-recognition embeddings to the reference person.

Adaptation of the metric in Zakharov et al. 2019, arXiv:1905.08233, reported by SadTalker,
arXiv:2211.12194, and the talking-head literature after it: the cosine
similarity between the identity embedding of the reference face image and the
identity embedding of the largest face in each generated frame, averaged over
frames. The embedding is produced by a face-recognition network; here the
InsightFace ``buffalo_l`` pack (SCRFD-10GF detector, ArcFace
ResNet50@WebFace600K recogniser) is used. This changes the recognition space
from the cited evaluations; absolute scores are not interchangeable.

Protocol:

* ``FaceAnalysis`` with the ``buffalo_l`` pack at its default ``det_size``;
  only detection and recognition models are loaded;
* the face with the largest bounding box per image; when no face is found the
  image is padded by 25% with gray (128) and detection is retried (the same
  retry the ConsisID/OpenS2V FaceSim scripts use);
* raw ArcFace cosine similarity per frame — no clamping; frames without a face
  are skipped; the video score is the mean over detected frames;
* ``reference_path`` may point to one face image or a directory of face images
  of the person (their embeddings are averaged into one reference vector, then
  renormalised).

Inputs: ``sample.path`` — video or image; ``sample.reference_path`` — face
image or directory of face images. Without a reference the metric is left unset.

csim — higher = more similar identity (raw cosine, -1 to 1).

Source: Zakharov et al., "Few-Shot Adversarial Learning of Realistic Neural
Talking Head Models", ICCV 2019, arXiv:1905.08233; SadTalker (OpenTalker),
arXiv:2211.12194. InsightFace buffalo_l models are released for non-commercial
research use only.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_WEIGHTS_REPO = "BestWishYsh/OpenS2V-Weight"
_BUFFALO = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class CSIMModule(PipelineModule):
    name = "csim"
    provenance = {
        "csim": "adapted",
        "csim_face_frames": "utility",
    }
    sources = {
        "csim": "Zakharov et al. 2019 (arXiv:1905.08233); SadTalker (arXiv:2211.12194); InsightFace ArcFace — https://github.com/deepinsight/insightface",
    }
    deviations = {
        "csim": "Uses InsightFace buffalo_l ArcFace embeddings instead of the cited evaluations' recognition networks; gray-padding detection retry, skipped undetected frames, and optional averaging of reference-image embeddings are Ayase protocol choices. Absolute scores are not comparable to those evaluations.",
    }
    description = "CSIM: mean ArcFace cosine similarity to the reference face (Zakharov 2019, SadTalker)"
    default_config = {
        "max_frames": 0,          # 0 = every frame (published protocol evaluates all frames)
        "det_size": 640,          # FaceAnalysis detection size
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {"id": "BestWishYsh/OpenS2V-Weight", "type": "huggingface",
         "url": "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/face_extractor/models/buffalo_l/det_10g.onnx",
         "task": "InsightFace buffalo_l SCRFD-10GF face detector", "size": "17 MB", "auto_download": True,
         "notes": "InsightFace model: non-commercial research use only"},
        {"id": "BestWishYsh/OpenS2V-Weight", "type": "huggingface",
         "url": "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/face_extractor/models/buffalo_l/w600k_r50.onnx",
         "task": "InsightFace buffalo_l ArcFace R50 identity embedding", "size": "174 MB", "auto_download": True,
         "notes": "InsightFace model: non-commercial research use only"},
    ]
    metric_info = {
        "csim": "CSIM: mean ArcFace cosine similarity to the reference face over detected frames (higher=better)",
        "csim_face_frames": "Frames with a detected face / evaluated frames (0-1)",
    }
    metric_groups = {
        "csim": "face",
        "csim_face_frames": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = int(self.config.get("max_frames", 0))
        self.det_size = int(self.config.get("det_size", 640))
        self._arc = None
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch  # noqa: F401
            from insightface.app import FaceAnalysis
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("csim: insightface unavailable (%s); CSIM left unset", e)
            return
        device = str(self.config.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            root = self._buffalo_root()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
                else ["CPUExecutionProvider"]
            self._arc = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers,
                                     allowed_modules=["detection", "recognition"])
            self._arc.prepare(ctx_id=0 if device.startswith("cuda") else -1,
                              det_size=(self.det_size, self.det_size))
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("csim: model setup failed (%s); CSIM left unset", e)
            return
        self._backend = "insightface"

    def _fetch(self, name: str, digest: str) -> Path:
        import hashlib

        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=_WEIGHTS_REPO, filename=name,
                                    cache_dir=str(Path(self.config.get("models_dir", "models")) / "csim")))
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise RuntimeError(f"{name} sha256 {got} != {digest}")
        return path

    def _buffalo_root(self) -> Path:
        """InsightFace root holding models/buffalo_l/ with the two downloaded files."""
        import shutil

        root = Path(self.config.get("models_dir", "models")).resolve() / "csim" / "insightface"
        pack = root / "models" / "buffalo_l"
        pack.mkdir(parents=True, exist_ok=True)
        for name, digest in _BUFFALO.items():
            target = pack / name
            if not target.exists():
                shutil.copyfile(self._fetch(f"face_extractor/models/buffalo_l/{name}", digest), target)
        return root

    # -- embedding -------------------------------------------------------------

    def _largest_face(self, image_bgr):
        faces = self._arc.get(image_bgr)
        if len(faces) > 0:
            return sorted(
                faces,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            )[-1]
        return None

    def _embedding(self, image_rgb) -> Optional[np.ndarray]:
        import cv2

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face = self._largest_face(image_bgr)
        if face is None:
            h, w = image_bgr.shape[:2]
            top, left = int(h * 0.25), int(w * 0.25)
            padded = cv2.copyMakeBorder(
                image_bgr, top, top, left, left, cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            face = self._largest_face(padded)
            if face is None:
                return None
        emb = face["embedding"]
        return emb / np.linalg.norm(emb)

    def _reference_embedding(self, ref: Path) -> Optional[np.ndarray]:
        from PIL import Image

        if ref.is_dir():
            images = sorted(p for p in ref.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        elif ref.suffix.lower() in _IMAGE_SUFFIXES:
            images = [ref]
        else:
            logger.warning("csim: reference must be a face image or a directory of face images, got %s", ref.name)
            return None
        embs = []
        for path in images:
            emb = self._embedding(np.array(Image.open(path).convert("RGB")))
            if emb is not None:
                embs.append(emb)
        if not embs:
            logger.warning("csim: no face found in reference %s; CSIM left unset", ref)
            return None
        mean = np.mean(embs, axis=0)
        return mean / np.linalg.norm(mean)

    def _frames(self, path: Path) -> List[np.ndarray]:
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            from PIL import Image

            return [np.array(Image.open(path).convert("RGB"))]
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        if self.max_frames > 0 and total > self.max_frames:
            idx = np.linspace(0, total - 1, self.max_frames, dtype=int)
            frames = []
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    @staticmethod
    def _cosine(a, b) -> float:
        return float(np.dot(a, b))

    def process(self, sample: Sample) -> Sample:
        if self._backend in (None, "unavailable") or self._arc is None:
            return sample
        ref = getattr(sample, "reference_path", None)
        if ref is None:
            return sample
        try:
            ref_emb = self._reference_embedding(Path(ref))
            if ref_emb is None:
                return sample
            frames = self._frames(Path(sample.path))
            if not frames:
                return sample
            scores, with_face = [], 0
            for frame in frames:
                emb = self._embedding(frame)
                if emb is None:
                    continue
                with_face += 1
                scores.append(self._cosine(ref_emb, emb))
        except Exception as e:
            logger.warning("csim failed on %s: %s", Path(sample.path).name, e)
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if scores:
            sample.quality_metrics.csim = float(np.mean(scores))
        sample.quality_metrics.csim_face_frames = with_face / len(frames)
        return sample
