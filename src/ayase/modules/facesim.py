"""FaceSim-Cur and FaceSim-Arc: face identity of a video against a reference face image.

Published metrics (ConsisID, Yuan et al., arXiv:2411.17440; carried into OpenS2V-Eval,
arXiv:2505.20292): the cosine similarity between the face in a real reference image and the
face in frames of a generated video, in the CurricularFace (FaceSim-Cur) and ArcFace
(FaceSim-Arc) feature spaces. This module is a port of the official evaluation scripts, not a
re-derivation; each step below names the line it follows.

Protocol ``consisid`` (default) follows PKU-YuanGroup/ConsisID ``eval/get_facesim_fid.py``:

* InsightFace ``FaceAnalysis`` with the ``buffalo_l`` pack (SCRFD-10GF detector, ArcFace
  ResNet50@WebFace600K recogniser), ``det_size=(320, 320)``;
* the face with the largest bounding box; if none is found, the image is padded by 25% on each
  side with gray (128) and detection is retried, keypoints shifted back;
* FaceSim-Arc embedding: the recogniser's ``embedding`` of that face;
* FaceSim-Cur: ``face_align.norm_crop(bgr, kps, image_size=224)`` -> RGB -> resize to 112 ->
  ``(x/255 - 0.5)/0.5`` -> CurricularFace IR-101 (``glint360k_curricular_face_r101_backbone.bin``)
  -> L2-normalised embedding;
* 16 frames at ``np.linspace(0, N-1, 16)``, read with OpenCV by seeking;
* per frame ``max(0, cos)``; frames without a face are skipped; the video score is the mean over
  frames, and 0.0 when no frame has a face (as in the official script).

Protocol ``opens2v`` follows PKU-YuanGroup/OpenS2V-Nexus ``eval/get_facesim.py`` (v1.1): 32 frames
read with decord; frames whose FaceSim-Cur is 0 are dropped from the Cur mean while FaceSim-Arc
keeps them (the official script compares the list, not the score, to 0.0 - reproduced on purpose so
that numbers match the published OpenS2V results). Reported values are not clipped at 1; OpenS2V's
``merge_result.py`` applies ``min(v, 1)`` when aggregating a benchmark.

Inputs: ``sample.path`` - video (or image); ``sample.reference_path`` - an image of the real
person. A video reference is rejected: the published metric is defined against a face image, so a
frame must be chosen by the caller.

Weights come from the Hugging Face repository the official scripts use,
``BestWishYsh/OpenS2V-Weight`` (Apache-2.0), and are checked by sha256: CurricularFace
``glint360k_curricular_face_r101_backbone.bin`` (identical by sha256 to ModelScope
``damo/cv_ir101_facerecognition_cfglint``) and the ``buffalo_l`` detector ``det_10g.onnx`` and
recogniser ``w600k_r50.onnx`` from ``face_extractor/models/buffalo_l/``. Only detection and
recognition are loaded; the other models of the pack take no part in the metric. InsightFace models
are released for non-commercial research use only.

Validation (official demo, OpenS2V ``eval/demo_result``): ``singleface_3.mp4`` against
``Images/singleface/crop_man/11.jpg`` with protocol ``opens2v``; published FaceSim-Cur
0.9420197624713182, FaceSim-Arc 0.9367029462009668. Reproduced 2026-09-24 (insightface 1.0.1,
onnxruntime-gpu 1.22.0, torch 2.10.0, decord 0.6.0, scikit-image 0.26.0): on GPU Cur 0.941648,
Arc 0.936496; on CPU 0.941843, 0.936898. The published Arc value lies between the GPU and CPU runs,
so the 2-4e-4 difference is the onnxruntime execution provider, not the protocol; TF32 on or off
changes Cur by 2e-6. Protocol ``consisid`` on the same pair: Cur 0.940112, Arc 0.936397.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_WEIGHTS_REPO = "BestWishYsh/OpenS2V-Weight"
_CUR_FILE = "glint360k_curricular_face_r101_backbone.bin"
_CUR_SHA256 = "aa54ae6dc9f7ac6e262809f1a81d6108e221952bf836aee4cb677ef19246c6ec"
_BUFFALO = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
_PROTOCOLS = {"consisid": 16, "opens2v": 32}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FaceSimModule(PipelineModule):
    name = "facesim"
    description = "FaceSim-Cur / FaceSim-Arc face identity vs a reference image (ConsisID, OpenS2V)"
    default_config = {
        "protocol": "consisid",   # "consisid" (16 frames, OpenCV) or "opens2v" (32 frames, decord)
        "num_frames": None,       # None = the protocol's own frame count
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {"id": "BestWishYsh/OpenS2V-Weight", "type": "huggingface",
         "url": "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/glint360k_curricular_face_r101_backbone.bin",
         "task": "CurricularFace IR-101 face embedding (FaceSim-Cur)", "size": "261 MB", "auto_download": True,
         "notes": "Apache-2.0 (ConsisID / OpenS2V weight repository); sha256 aa54ae6d..."},
        {"id": "BestWishYsh/OpenS2V-Weight", "type": "huggingface",
         "url": "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/face_extractor/models/buffalo_l/det_10g.onnx",
         "task": "InsightFace buffalo_l SCRFD-10GF face detector", "size": "17 MB", "auto_download": True,
         "notes": "InsightFace model: non-commercial research use only"},
        {"id": "BestWishYsh/OpenS2V-Weight", "type": "huggingface",
         "url": "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/face_extractor/models/buffalo_l/w600k_r50.onnx",
         "task": "InsightFace buffalo_l ArcFace R50 embedding (FaceSim-Arc)", "size": "174 MB", "auto_download": True,
         "notes": "InsightFace model: non-commercial research use only"},
    ]
    metric_info = {
        "facesim_cur": "FaceSim-Cur: CurricularFace cosine to the reference face, mean over frames (higher=better)",
        "facesim_arc": "FaceSim-Arc: ArcFace (buffalo_l) cosine to the reference face, mean over frames (higher=better)",
        "facesim_face_frames": "Frames with a detected face / sampled frames (0-1)",
    }
    metric_groups = {
        "facesim_cur": "face",
        "facesim_arc": "face",
        "facesim_face_frames": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.protocol = str(self.config.get("protocol", "consisid")).lower()
        if self.protocol not in _PROTOCOLS:
            raise ValueError(f"facesim: protocol must be one of {sorted(_PROTOCOLS)}, got {self.protocol!r}")
        self.num_frames = int(self.config.get("num_frames") or _PROTOCOLS[self.protocol])
        self._backend = None
        self._arc = None
        self._cur = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from insightface.app import FaceAnalysis

            from ayase.vendor.curricularface import get_model
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("facesim: dependencies missing (%s); FaceSim left unset", e)
            return
        if self.protocol == "opens2v":
            try:
                import decord  # noqa: F401
            except Exception as e:
                self._backend = "unavailable"
                logger.warning("facesim: protocol opens2v needs decord (%s); FaceSim left unset", e)
                return

        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        try:
            weight = self._fetch(_CUR_FILE, _CUR_SHA256)
            root = self._buffalo_root()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
                else ["CPUExecutionProvider"]
            self._arc = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers,
                                     allowed_modules=["detection", "recognition"])
            self._arc.prepare(ctx_id=0 if device.startswith("cuda") else -1, det_size=(320, 320))
            model = get_model("IR_101")([112, 112])
            model.load_state_dict(torch.load(str(weight), map_location="cpu"))
            self._cur = model.to(device).eval()
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("facesim: model setup failed (%s); FaceSim left unset", e)
            return
        self._backend = "consisid" if self.protocol == "consisid" else "opens2v"

    def _fetch(self, name: str, digest: str) -> Path:
        import hashlib

        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=_WEIGHTS_REPO, filename=name,
                                    cache_dir=str(Path(self.config.get("models_dir", "models")) / "facesim")))
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise RuntimeError(f"{name} sha256 {got} != {digest}")
        return path

    def _buffalo_root(self) -> Path:
        """InsightFace root holding models/buffalo_l/ with the two downloaded files."""
        import shutil

        root = Path(self.config.get("models_dir", "models")).resolve() / "facesim" / "insightface"
        pack = root / "models" / "buffalo_l"
        pack.mkdir(parents=True, exist_ok=True)
        for name, digest in _BUFFALO.items():
            target = pack / name
            if not target.exists():
                shutil.copyfile(self._fetch(f"face_extractor/models/buffalo_l/{name}", digest), target)
        return root

    # -- official steps ---------------------------------------------------------------------

    def _largest_face(self, image_bgr):
        faces = self._arc.get(image_bgr)
        if len(faces) > 0:
            return sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        return None

    def _process_image(self, image_rgb) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import cv2
        from insightface.utils import face_align

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face_info = self._largest_face(image_bgr)
        if face_info is None:
            h, w = image_bgr.shape[:2]
            top, left = int(h * 0.25), int(w * 0.25)
            padded = cv2.copyMakeBorder(image_bgr, top, top, left, left, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            face_info = self._largest_face(padded)
            if face_info is None:
                return None, None
            face_kps = face_info["kps"] - np.array((left, top))
        else:
            face_kps = face_info["kps"]
        norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
        return cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB), face_info["embedding"]

    def _cur_embedding(self, align_face_rgb) -> np.ndarray:
        import cv2
        import torch

        img = cv2.resize(align_face_rgb, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self._device)
        img.div_(255).sub_(0.5).div_(0.5)
        with torch.no_grad():
            embedding = self._cur(img).detach().cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def _cosine(a, b) -> float:
        import torch

        return float(torch.nn.functional.cosine_similarity(torch.tensor(a), torch.tensor(b), dim=-1).item())

    def _frames(self, path: Path) -> List[np.ndarray]:
        """RGB frames at the protocol's indices."""
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            from PIL import Image

            return [np.array(Image.open(path).convert("RGB"))]
        if self.protocol == "opens2v":
            import decord

            vr = decord.VideoReader(str(path))
            idx = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
            return [vr[int(i)].asnumpy() for i in idx]
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in np.linspace(0, total - 1, self.num_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def process(self, sample: Sample) -> Sample:
        if self._backend in (None, "unavailable") or self._arc is None:
            return sample
        ref = getattr(sample, "reference_path", None)
        if ref is None:
            return sample
        ref = Path(ref)
        if ref.suffix.lower() not in _IMAGE_SUFFIXES:
            logger.warning("facesim: reference must be a face image, got %s; FaceSim left unset", ref.name)
            return sample
        try:
            from PIL import Image

            ref_face, ref_arc = self._process_image(np.array(Image.open(ref).convert("RGB")))
            if ref_face is None:
                logger.warning("facesim: no face in reference %s; FaceSim left unset", ref.name)
                return sample
            ref_cur = self._cur_embedding(ref_face)

            frames = self._frames(Path(sample.path))
            cur_scores, arc_scores, with_face = [], [], 0
            for frame in frames:
                face, arc = self._process_image(frame)
                if face is None:
                    continue
                with_face += 1
                cur = max(0.0, self._cosine(ref_cur, self._cur_embedding(face)))
                arc_s = max(0.0, self._cosine(ref_arc, arc))
                if self.protocol == "opens2v":
                    if cur != 0.0:
                        cur_scores.append(cur)
                    arc_scores.append(arc_s)  # official v1.1 keeps zero Arc scores (see docstring)
                else:
                    cur_scores.append(cur)
                    arc_scores.append(arc_s)
        except Exception as e:
            logger.warning("facesim failed on %s: %s", Path(sample.path).name, e)
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.facesim_cur = float(np.mean(cur_scores)) if cur_scores else 0.0
        sample.quality_metrics.facesim_arc = float(np.mean(arc_scores)) if arc_scores else 0.0
        sample.quality_metrics.facesim_face_frames = with_face / len(frames) if frames else 0.0
        return sample
