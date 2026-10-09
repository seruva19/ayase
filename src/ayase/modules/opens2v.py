"""OpenS2V-Eval subject-consistency metrics (NexusScore + NaturalScore).

Implements the two subject-driven video-generation metrics from OpenS2V-Nexus
(PKU-YuanGroup, arXiv:2505.20292, ``eval/get_nexusscore.py`` /
``eval/get_naturalscore.py``):

* **NexusScore** ``opens2v_nexus_score`` — subject consistency. The canonical
  backend (``nexus_backend="opens2v"``) reproduces the upstream pipeline
  verbatim: a **YOLO-World v2-L image-prompt adapter**
  (``yolo_world_v2_l_image_prompt_adapter-719a7afb.pth`` via the mmyolo/MMEngine
  runner) localizes the subject in 32 uniformly sampled frames, conditioned by
  the *reference image* embedding (CLIP-ViT-B/32 vision encoder -> the
  adapter's projector), with ``score_thr=0.5`` / ``nms_thr=0.7`` /
  ``max_num_boxes=100``. Detected crops and the reference image are embedded
  with **GME-Qwen2-VL-7B** (``Alibaba-NLP/gme-Qwen2-VL-7B-Instruct``); kept
  detections satisfy ``bbox_conf > 0.6`` and ``gme_text_score > 0.30``;
  the score is ``mean(kept image scores) / frame_obj`` (``frame_obj`` = frames
  containing at least one detection), exactly as upstream.

  ``nexus_backend="gdino"`` is an opt-in documented deviation (GroundingDINO
  text-prompted detector + CLIP/DINOv2 crop encoder); ``"auto"`` prefers the
  canonical backend and falls back to gdino when the mmyolo/yolo_world stack
  is not installed. The canonical stack requires
  ``pip install mmyolo`` plus the ``yolo_world`` package
  (``pip install git+https://github.com/AILab-CVC/YOLO-World``).

* **NaturalScore** ``opens2v_natural_score`` — naturalness. The canonical
  backend (``natural_judge="openai"``) reproduces the upstream judge verbatim:
  16 frames sampled at stride ``total//16``, each resized to long-side 512 and
  sent as base64 JPEG to ``gpt-4o-2024-11-20`` with the upstream rubric prompt;
  three independent runs are averaged. Requires an OpenAI API key (config
  ``openai_api_key`` or the ``OPENAI_API_KEY`` env var). ``natural_judge="vlm"``
  is an opt-in local-VLM deviation (per-frame rubric, averaged); ``"auto"``
  prefers OpenAI when a key is configured and falls back to the local VLM.

STRICT real-or-none policy: there is **no** whole-frame-similarity fallback. If
the detector/encoder stack is unavailable, ``opens2v_nexus_score`` stays
``None``; if no judge is available, ``opens2v_natural_score`` stays ``None``.

Reference subject image (precedence): ``sample.reference_path`` (an image file,
or a directory whose images are averaged), else config ``reference_image``.
Subject phrase (precedence): config ``subject_prompt``, else
``sample.caption.text``. Upstream supports multiple reference images/labels per
video; Ayase's generic sample model carries a single reference pair, which is
the documented adaptation.
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ayase.image import IMAGE_EXTENSIONS, load_pil_image, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# DINOv2 backbone weights (mirror, consistent with dino_face_identity/i2v_similarity).
_DINOV2_MIRROR_BASE = "https://huggingface.co/AkaneTendo25/ayase-assets/resolve/main/"
_DINOV2_WEIGHTS = {
    "dinov2_vitb14": "dino_face_identity/dinov2_vitb14_pretrain.pth",
}

# OpenS2V NaturalScore judge prompt (verbatim, eval/get_naturalscore.py).
_NATURAL_PROMPT = """
Your task is to determine how realistic the given video clip appears, based on 16 extracted frames. Consider the following aspects in your evaluation:

- **Common sense consistency**: Are the objects, people, and interactions logically coherent in the context of the video?
- **Physical plausibility**: Do lighting, shadows, motion, and reflections obey the laws of physics? Are the objects in motion consistent with real-world physics?
- **Naturalness**: Does the visual quality (textures, details, proportions, etc.) resemble what we would expect in real life? Is there any unnatural visual distortion?
- **AI generation artifacts**: Are there signs of unnatural blurring, morphing, glitches, distortions, or inconsistencies across frames?

**If the video contains humans**, pay special attention to:
- Are the facial features realistic and anatomically correct (e.g., eyes, mouth, and nose proportions)?
- Do the body parts appear proportionate and natural in motion (e.g., arm and leg movements, hand gestures)?

If **no humans** are present in the video, you can focus on evaluating the realism of other visual aspects like object consistency, motion fluidity, and environmental plausibility without needing to specifically assess human-related elements.

Output a score from 1 to 5 based on the criteria below, followed by an explanation of the reasoning behind your score:

- **1 — Definitely AI-Generated**: Clear and frequent artifacts (e.g., blurry faces or objects, unnatural movements, inconsistent lighting), distorted shapes, implausible physics (e.g., impossible movements, lighting issues), and severe inconsistencies. Violates common sense or real-world logic. Faces and bodies may be unrealistic or distorted if humans are present.
- **2 — Likely AI-Generated**: Noticeable AI generation cues such as inconsistent anatomy, fluctuating object textures, or mild physical implausibility (e.g., unnatural hand positions or eye movements). Faces and bodies may appear unnatural or inconsistent if humans are present. Still clearly synthetic upon inspection.
- **3 — Uncertain / Borderline**: Mixed indicators — the video may appear mostly natural but contains subtle flaws or small anomalies that raise suspicion. Faces and bodies might show mild inconsistencies (e.g., slight distortion in facial features or body parts) if humans are present. Hard to determine definitively.
- **4 — Likely Real**: Mostly natural and physically plausible, with only minor and rare irregularities that might be explainable (e.g., slight compression, mild lighting inconsistencies). Faces and body parts are mostly natural, with only minor imperfections, if humans are present.
- **5 — Definitely Real**: Fully consistent with real-world physics, common sense, and appearance. No visible artifacts or signs of AI generation. Faces and body parts appear fully realistic, without any visible distortions or unnatural movements, if humans are present.

Please only return the score (1-5), no additional explanation.
"""

# Upstream OpenS2V-Weight checkpoint for the image-prompt YOLO-World adapter.
_YOLO_CKPT_NAME = "yolo_world_v2_l_image_prompt_adapter-719a7afb.pth"
_YOLO_CKPT_URL = "https://huggingface.co/BestWishYsh/OpenS2V-Weight/resolve/main/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth"
_YOLO_CFG_REL = os.path.join(
    "configs", "yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py"
)


def _mmyolo_config_path(rel_path: str) -> Optional[str]:
    """Resolve an mmyolo repo-relative config inside the installed package."""
    try:
        import mmyolo
    except ImportError:
        return None
    base = os.path.dirname(mmyolo.__file__)
    for root in (os.path.join(base, ".mim"), base):
        candidate = os.path.join(root, rel_path.replace("/", os.sep))
        if os.path.isfile(candidate):
            return candidate
    return None


class OpenS2VModule(PipelineModule):
    name = "opens2v"
    provenance = "adapted"
    sources = {
        "opens2v_natural_score": "OpenS2V-Nexus NaturalScore (arXiv 2505.20292) — https://github.com/PKU-YuanGroup/OpenS2V-Nexus",
        "opens2v_nexus_score": "OpenS2V-Nexus NexusScore (arXiv 2505.20292) — https://github.com/PKU-YuanGroup/OpenS2V-Nexus",
    }
    deviations = {
        "opens2v_natural_score": "canonical 'openai' replicates upstream (GPT-4o x3 over 16 stride-sampled frames, verbatim prompt); 'vlm' is a local VLM judge substitute (documented substitution)",
        "opens2v_nexus_score": "canonical 'opens2v' replicates upstream (YOLO-World image-prompt + GME-Qwen2-VL-7B); 'gdino' substitutes GroundingDINO+CLIP/DINOv2 for the detector/encoder (documented substitution). A single reference pair instead of upstream's img_paths×labels list",
    }
    description = (
        "OpenS2V-Eval subject-consistency metrics: NexusScore (YOLO-World image-prompt "
        "subject crops vs reference subject image, GME embeddings) and NaturalScore "
        "(GPT-4o naturalness judge)"
    )
    default_config = {
        "device": "auto",
        "nexus_backend": "auto",    # "auto" | "opens2v" | "gdino"
        "natural_judge": "auto",    # "auto" | "openai" | "vlm"
        # Canonical NexusScore stack (upstream eval/get_nexusscore.py)
        "nexus_frames": 32,          # upstream: 32 linspace frames
        "yolo_checkpoint": _YOLO_CKPT_NAME,
        "yolo_clip_model": "openai/clip-vit-base-patch32",
        "gme_model": "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        "det_score_thr": 0.5,        # upstream yoloworld_inference score_thr
        "det_nms_thr": 0.7,          # upstream yoloworld_inference nms_thr
        "det_max_boxes": 100,        # upstream max_num_boxes
        "keep_box_conf": 0.6,        # upstream aggregation gate
        "keep_text_sim": 0.30,       # upstream aggregation gate
        # GroundingDINO fallback detector (opt-in deviation)
        "detector_model": "IDEA-Research/grounding-dino-tiny",
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "gdino_keep_box_conf": 0.30,
        "gdino_keep_text_sim": 0.20,
        "encoder": "clip",           # "clip" (CLIP-I) | "dino" (DINOv2)
        "clip_model": "openai/clip-vit-base-patch32",
        "dino_model": "dinov2_vitb14",
        "max_frames": 16,            # gdino path frame count
        # Canonical NaturalScore judge (upstream eval/get_naturalscore.py)
        "openai_model": "gpt-4o-2024-11-20",
        "openai_api_key": None,      # else OPENAI_API_KEY env var
        "openai_base_url": None,
        "natural_frames": 16,        # upstream: stride total//16
        "natural_runs": 3,           # upstream: naturalscore_1/2/3
        # Local VLM judge (opt-in deviation)
        "vlm_model": "llava-hf/llava-1.5-7b-hf",
        "vlm_max_frames": 4,
        "vlm_max_new_tokens": 8,
        # Reference / subject-phrase resolution
        "subject_prompt": None,  # explicit subject phrase; else sample.caption.text
        "reference_image": None,  # fallback subject image if sample.reference_path unset
        "warning_threshold": 0.0,
        "models_dir": "models",
    }
    metric_groups = {
        "opens2v_nexus_score": "i2v",
        "opens2v_natural_score": "nr_quality",
    }
    metric_info = {
        "opens2v_nexus_score": (
            "OpenS2V NexusScore — subject consistency of detected subject crops vs the "
            "reference subject image (higher=better)"
        ),
        "opens2v_natural_score": (
            "OpenS2V NaturalScore — VLM naturalness rating on the 1-5 anti-copy-paste "
            "rubric (higher=better)"
        ),
    }
    models = [
        {
            "id": "BestWishYsh/OpenS2V-Weight",
            "type": "huggingface",
            "task": "YOLO-World v2-L image-prompt subject detector (canonical)",
            "notes": f"file {_YOLO_CKPT_NAME}",
        },
        {
            "id": "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
            "type": "huggingface",
            "task": "GME image/text embeddings (canonical NexusScore)",
        },
        {
            "id": "openai/clip-vit-base-patch32",
            "type": "huggingface",
            "task": "YOLO-World prompt encoder / CLIP fallback encoder",
        },
        {
            "id": "gpt-4o-2024-11-20",
            "type": "other",
            "task": "NaturalScore judge via OpenAI API (canonical)",
        },
        {
            "id": "mmyolo+yolo_world",
            "type": "pip_package",
            "install": "pip install mmyolo && pip install git+https://github.com/AILab-CVC/YOLO-World",
            "task": "MMEngine YOLO-World runner stack (canonical NexusScore)",
        },
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._device = "cpu"
        self._encoder = str(self.config.get("encoder", "clip")).lower()

        # Backend handles
        self._gdino_model = None
        self._gdino_processor = None
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._dino_transform = None
        self._vlm_model = None
        self._vlm_processor = None
        self._yolo_runner = None
        self._yolo_vision = None
        self._yolo_processor = None
        self._yolo_txt_feats = None
        self._gme = None
        self._openai_client = None

        # Availability flags
        self._gdino_ok = False
        self._clip_ok = False
        self._dino_ok = False
        self._vlm_ok = False
        self._nexus_available = False
        self._nexus_backend = None
        self._natural_available = False
        self._natural_backend = None
        self._vlm_tag = "vlm"

        self._backend = "unavailable"

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        try:
            import torch  # noqa: F401
            from ayase.runtime import resolve_torch_device
        except ImportError:
            logger.warning("PyTorch not installed. OpenS2V disabled.")
            return

        self._device = resolve_torch_device(self.config.get("device", "auto"))

        nexus_backend = str(self.config.get("nexus_backend", "auto")).lower()
        natural_judge = str(self.config.get("natural_judge", "auto")).lower()

        # NexusScore: canonical upstream stack first, GDINO as opt-in/fallback.
        if nexus_backend in ("auto", "opens2v"):
            self._load_official_nexus()
        if self._nexus_backend is None and nexus_backend in ("auto", "gdino"):
            self._load_detector()
            self._load_clip()
            if self._encoder == "dino":
                self._load_dino()
            if self._gdino_ok and self._clip_ok and (
                self._encoder != "dino" or self._dino_ok
            ):
                self._nexus_backend = "gdino"
        elif self._nexus_backend is None and nexus_backend == "opens2v":
            logger.warning(
                "OpenS2V: canonical nexus backend requested but mmyolo/yolo_world "
                "stack is unavailable; opens2v_nexus_score will not be populated"
            )
        self._nexus_available = self._nexus_backend is not None

        # NaturalScore: canonical GPT-4o judge first, local VLM as opt-in/fallback.
        if natural_judge in ("auto", "openai"):
            self._load_openai()
        if self._natural_backend is None and natural_judge in ("auto", "vlm"):
            self._load_vlm()
            if self._vlm_ok:
                self._natural_backend = "vlm"
        elif self._natural_backend is None and natural_judge == "openai":
            logger.warning(
                "OpenS2V: openai judge requested but no API key configured; "
                "opens2v_natural_score will not be populated"
            )
        self._natural_available = self._natural_backend is not None

        parts: List[str] = []
        if self._nexus_backend:
            parts.append(self._nexus_backend)
        if self._natural_backend:
            parts.append(self._natural_backend if self._natural_backend != "vlm" else self._vlm_tag)
        self._backend = "+".join(parts) if parts else "unavailable"

        logger.info(
            "OpenS2V ready: backend=%s nexus=%s natural=%s",
            self._backend,
            self._nexus_backend,
            self._natural_backend,
        )

    # -- canonical NexusScore loaders ---------------------------------------

    def _load_official_nexus(self) -> None:
        """Load the upstream YOLO-World image-prompt runner + GME encoder."""
        try:
            runner, vision_model, vision_processor, txt_feats = self._build_yolo_runner()
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: canonical YOLO-World stack unavailable: %s", e)
            runner = None
        if runner is None:
            return
        try:
            from ayase.third_party.opens2v.gme_model import GmeQwen2VL
            from ayase.runtime import shared_runtime_resource

            gme_path = self.config.get("gme_model", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
            device = self._device

            def load_gme():
                return GmeQwen2VL(model_path=gme_path, device=device)

            self._gme = shared_runtime_resource(self, ("gme_qwen2vl", gme_path, device), load_gme)
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: GME encoder unavailable: %s", e)
            return

        self._yolo_runner, self._yolo_vision, self._yolo_processor, self._yolo_txt_feats = (
            runner, vision_model, vision_processor, txt_feats,
        )
        self._nexus_backend = "opens2v"

    def _build_yolo_runner(self):
        """Upstream ``load_model_and_config``: Runner + CLIP prompt encoders."""
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmengine.runner import Runner
        from mmyolo.registry import RUNNERS
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        base_cfg = _mmyolo_config_path(
            "configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py"
        )
        if base_cfg is None:
            raise ImportError("mmyolo base config not found (pip install mmyolo)")

        # The vendored config's ``_base_`` is repo-relative; rewrite it to the
        # absolute path inside the installed mmyolo package.
        vendored = Path(__file__).resolve().parent.parent / "third_party" / "opens2v" / _YOLO_CFG_REL
        text = vendored.read_text(encoding="utf-8")
        text = re.sub(
            r'^_base_\s*=\s*["\'][^"\']*["\']',
            f'_base_ = "{base_cfg.replace(os.sep, "/")}"',
            text,
            count=1,
            flags=re.M,
        )
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix="_yoloworld_cfg.py", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()

        from ayase.config import download_model_file

        models_dir = str(self.config.get("models_dir", "models"))
        ckpt = download_model_file(
            f"opens2v/{self.config.get('yolo_checkpoint', _YOLO_CKPT_NAME)}",
            _YOLO_CKPT_URL,
            models_dir,
        )
        clip_model_path = self.config.get("yolo_clip_model", "openai/clip-vit-base-patch32")
        device = self._device

        cfg = Config.fromfile(tmp.name)
        cfg.load_from = str(ckpt)
        cfg.text_model_name = clip_model_path
        cfg.model.vision_model = clip_model_path
        cfg.model.backbone.text_model.model_name = clip_model_path

        runner = (
            Runner.from_cfg(cfg)
            if "runner_type" not in cfg
            else RUNNERS.build(cfg)
        )
        runner.call_hook("before_run")
        runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = "mmdet.LoadImageFromNDArray"
        runner.pipeline = Compose(pipeline)
        runner.model.eval()

        vision_processor = AutoProcessor.from_pretrained(clip_model_path)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
        vision_model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(clip_model_path, use_fast=True)
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
        text_model.to(device)

        texts = tokenizer(text=[" "], return_tensors="pt", padding=True).to(device)
        txt_feats = text_model(**texts).text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])[0].unsqueeze(0)

        return runner, vision_model, vision_processor, txt_feats

    # -- canonical NaturalScore loader ----------------------------------------

    def _load_openai(self) -> None:
        api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.info("OpenS2V: no OpenAI API key (openai_api_key / OPENAI_API_KEY)")
            return
        try:
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=api_key, base_url=self.config.get("openai_base_url")
            )
            self._natural_backend = "openai"
            logger.info("OpenS2V: OpenAI judge ready (%s)", self.config.get("openai_model"))
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: OpenAI client unavailable: %s", e)

    # -- deviation backend loaders --------------------------------------------

    def _load_detector(self) -> None:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                shared_runtime_resource,
            )

            model_id = self.config.get("detector_model", "IDEA-Research/grounding-dino-tiny")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_id, models_dir)
            device = self._device

            def load() -> Tuple[Any, Any]:
                model = (
                    from_pretrained_with_attention(
                        AutoModelForZeroShotObjectDetection,
                        resolved,
                        self.config,
                        device=device,
                    )
                    .to(device)
                    .eval()
                )
                processor = AutoProcessor.from_pretrained(resolved)
                return model, processor

            self._gdino_model, self._gdino_processor = shared_runtime_resource(
                self, ("gdino", resolved, str(device)), load
            )
            self._gdino_ok = True
            logger.info("OpenS2V: GroundingDINO loaded (%s) on %s", model_id, device)
        except Exception as e:  # noqa: BLE001 — real-or-none: any failure => detector off
            logger.info("OpenS2V: GroundingDINO unavailable: %s", e)

    def _load_clip(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                shared_runtime_resource,
            )

            name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(name, models_dir)
            device = self._device

            def load() -> Tuple[Any, Any]:
                model = (
                    from_pretrained_with_attention(
                        CLIPModel, resolved, self.config, device=device
                    )
                    .to(device)
                    .eval()
                )
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            # Same shared key layout as concept_presence so the CLIP backbone is reused.
            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load,
            )
            self._clip_ok = True
            logger.info("OpenS2V: CLIP loaded (%s) on %s", name, device)
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: CLIP unavailable: %s", e)

    def _load_dino(self) -> None:
        try:
            import torch
            from torchvision import transforms as T

            model_name = self.config.get("dino_model", "dinov2_vitb14")
            models_dir = self.config.get("models_dir", "models")
            rel = _DINOV2_WEIGHTS.get(model_name)

            model = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=(rel is None)
            )
            if rel:
                from ayase.config import download_model_file, resolve_assets_url

                ckpt = download_model_file(
                    rel,
                    resolve_assets_url(_DINOV2_MIRROR_BASE + rel, self.config),
                    models_dir,
                )
                model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            model.eval().to(self._device)

            self._dino_model = model
            self._dino_transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self._dino_ok = True
            logger.info("OpenS2V: DINOv2 loaded (%s) on %s", model_name, self._device)
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: DINOv2 unavailable: %s", e)

    def _load_vlm(self) -> None:
        try:
            import torch
            # The default checkpoint (llava-hf/llava-1.5-7b-hf) is a LLaVA-1.5
            # model, NOT LLaVA-NeXT — loading it with the LlavaNext* classes
            # raises and disables NaturalScore. Use the version-robust
            # AutoModelForImageTextToText + AutoProcessor, which resolve to the
            # correct Llava*/LlavaNext* implementation from the checkpoint config
            # (verified against the llava-1.5-7b-hf model card).
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )
            from ayase.runtime import (
                from_pretrained_with_attention,
                shared_runtime_resource,
            )

            name = self.config.get("vlm_model", "llava-hf/llava-1.5-7b-hf")
            models_dir = self.config.get("models_dir", "models")
            device = self._device
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

            def load() -> Tuple[Any, Any]:
                model = (
                    from_pretrained_with_attention(
                        AutoModelForImageTextToText,
                        name,
                        self.config,
                        device=device,
                        torch_dtype=dtype,
                        cache_dir=models_dir,
                        low_cpu_mem_usage=True,
                    )
                    .to(device)
                    .eval()
                )
                processor = AutoProcessor.from_pretrained(name, cache_dir=models_dir)
                return model, processor

            self._vlm_model, self._vlm_processor = shared_runtime_resource(
                self, ("opens2v_vlm", name, str(device)), load
            )
            self._vlm_ok = True
            self._vlm_tag = name.split("/")[-1].split("-")[0] or "vlm"
            logger.info("OpenS2V: VLM loaded (%s) on %s", name, device)
        except Exception as e:  # noqa: BLE001
            logger.info("OpenS2V: VLM unavailable: %s", e)

    # ------------------------------------------------------------------ #
    #  Processing                                                         #
    # ------------------------------------------------------------------ #

    def process(self, sample: Sample) -> Sample:
        if not (self._nexus_available or self._natural_available):
            return sample

        try:
            nexus: Optional[float] = None
            if self._nexus_available:
                nexus = self._compute_nexus(sample)

            natural: Optional[float] = None
            if self._natural_available:
                natural = self._compute_natural(sample)

            if nexus is None and natural is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            if nexus is not None:
                sample.quality_metrics.opens2v_nexus_score = round(float(nexus), 4)
            if natural is not None:
                sample.quality_metrics.opens2v_natural_score = round(float(natural), 4)

            warn = float(self.config.get("warning_threshold", 0.0))
            if nexus is not None and nexus <= warn:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"OpenS2V NexusScore is low ({nexus:.4f}): the reference "
                            "subject was not consistently detected across frames"
                        ),
                        details={"opens2v_nexus_score": nexus},
                        recommendation="Check the subject phrase and reference subject image.",
                    )
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenS2V failed for %s: %s", sample.path, e)

        return sample

    # ------------------------------------------------------------------ #
    #  NexusScore                                                         #
    # ------------------------------------------------------------------ #

    def _compute_nexus(self, sample: Sample) -> Optional[float]:
        phrase = self._resolve_phrase(sample)
        ref_path = self._resolve_reference(sample)
        if not phrase or ref_path is None:
            # No subject phrase or reference subject image => cannot run (strict None).
            return None

        if self._nexus_backend == "opens2v":
            return self._compute_nexus_official(sample, ref_path, phrase)
        return self._compute_nexus_gdino(sample, ref_path, phrase)

    # -- canonical upstream path ----------------------------------------------

    def _generate_prompt_embeddings(self, prompt_image: Image.Image):
        """Upstream ``generate_image_embeddings``: CLIP vision embed -> projector."""
        import torch

        prompt_image = prompt_image.convert("RGB")
        inputs = self._yolo_processor(
            images=[prompt_image], return_tensors="pt", padding=True
        ).to(self._device)
        image_outputs = self._yolo_vision(**inputs)
        img_feats = image_outputs.image_embeds.view(1, -1)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        projector = getattr(self._yolo_runner.model, "image_prompt_encoder", None)
        projector = getattr(projector, "projector", None) if projector is not None else None
        if projector is not None:
            img_feats = projector(img_feats)
        return img_feats

    def _yoloworld_inference(self, frame: Image.Image, prompt_image: Image.Image):
        """Upstream ``yoloworld_inference``: image-prompt-conditioned detection."""
        import torch
        from mmengine.runner.amp import autocast
        from torchvision.ops import nms

        image = frame.convert("RGB")
        prompt_embeddings = self._generate_prompt_embeddings(prompt_image)
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )
        runner = self._yolo_runner
        runner.model.num_test_classes = prompt_embeddings.shape[0]
        runner.model.setembeddings(prompt_embeddings[None])

        data_info = {"img_id": 0, "img": np.array(image), "texts": [["object"], [" "]]}
        data_info = runner.pipeline(data_info)
        data_batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }

        with autocast(enabled=False), torch.no_grad():
            if "texts" in data_batch["data_samples"][0]:
                del data_batch["data_samples"][0]["texts"]
            output = runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        keep = nms(
            pred_instances.bboxes,
            pred_instances.scores,
            iou_threshold=float(self.config.get("det_nms_thr", 0.7)),
        )
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[
            pred_instances.scores.float() > float(self.config.get("det_score_thr", 0.5))
        ]

        max_boxes = int(self.config.get("det_max_boxes", 100))
        if len(pred_instances.scores) > max_boxes:
            indices = pred_instances.scores.float().topk(max_boxes)[1]
            pred_instances = pred_instances[indices]

        return pred_instances.cpu().numpy()

    def _compute_nexus_official(
        self, sample: Sample, ref_path: Path, phrase: str
    ) -> Optional[float]:
        """Verbatim upstream scoring: image-prompt detections + GME + gates."""
        frames = sample_frames(
            sample.path,
            max_frames=int(self.config.get("nexus_frames", 32)),
            color="rgb",
        )
        if not frames:
            return None

        prompt_image = load_pil_image(ref_path)
        if prompt_image is None:
            return None

        all_local_images: List[Image.Image] = []
        all_yolo_conf: List[float] = []
        frame_obj = 0

        for arr in frames:
            frame = Image.fromarray(arr)
            pred = self._yoloworld_inference(frame, prompt_image)
            bboxes = pred["bboxes"]
            confidences = pred["scores"]
            all_yolo_conf.extend(confidences.tolist())

            if len(bboxes) != 0:
                frame_obj += 1
            for bbox in bboxes:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                all_local_images.append(
                    frame.crop((x1, y1, x2, y2))
                )

        if not all_local_images:
            return None

        import torch

        e_main = self._gme.get_image_embeddings(
            images=[prompt_image] * len(all_local_images),
            is_query=False,
            show_progress_bar=False,
        )
        e_query = self._gme.get_text_embeddings(
            texts=[phrase] * len(all_local_images),
            instruction="Find an image that matches the given text.",
            show_progress_bar=False,
        )
        e_local = self._gme.get_image_embeddings(
            images=all_local_images, is_query=False, show_progress_bar=False
        )

        gme_image_score = (e_main * e_local).sum(-1)
        gme_text_score = (e_query * e_local).sum(-1)

        kept: List[float] = []
        for bbox_conf, text_conf, nexus_score in zip(
            all_yolo_conf, gme_text_score, gme_image_score
        ):
            if (
                float(bbox_conf) > float(self.config.get("keep_box_conf", 0.6))
                and float(text_conf) > float(self.config.get("keep_text_sim", 0.30))
                and float(nexus_score) != 0
            ):
                kept.append(float(nexus_score))

        if kept:
            return float(torch.mean(torch.tensor(kept)).item()) / max(frame_obj, 1)
        return 0.0

    # -- GDINO deviation path --------------------------------------------------

    def _compute_nexus_gdino(
        self, sample: Sample, ref_path: Path, phrase: str
    ) -> Optional[float]:
        frames = sample_frames(
            sample.path,
            max_frames=int(self.config.get("max_frames", 16)),
            color="rgb",
        )
        if not frames:
            return None

        which = "dino" if self._encoder == "dino" else "clip"
        ref_emb = self._reference_embed(ref_path, which)
        if ref_emb is None:
            return None

        all_crops: List[Image.Image] = []
        all_confs: List[float] = []
        frame_obj = 0
        for frame in frames:
            pil = Image.fromarray(np.ascontiguousarray(frame))
            dets = self._detect_subject(pil, phrase)
            if dets:
                frame_obj += 1
            for crop, score in dets:
                all_crops.append(crop)
                all_confs.append(score)

        if not all_crops:
            # Detector ran but found no subject in any frame — a real 0 (repo behavior),
            # distinct from None (detector unavailable).
            return 0.0

        import torch

        clip_crop = self._clip_image_embeds(all_crops)  # [N, Dc] normalized (text gate)
        text_emb = self._clip_text_embed(phrase)  # [Dc]
        text_sims = (clip_crop @ text_emb).detach().cpu().tolist()

        if which == "dino":
            dino_crop = self._dino_image_embeds(all_crops)  # [N, Dd]
            image_sims = (dino_crop @ ref_emb).detach().cpu().tolist()
        else:
            image_sims = (clip_crop @ ref_emb).detach().cpu().tolist()

        detections = list(zip(all_confs, text_sims, image_sims))
        return self._aggregate_nexus(
            detections,
            frame_obj,
            box_conf_threshold=float(self.config.get("gdino_keep_box_conf", 0.30)),
            text_sim_threshold=float(self.config.get("gdino_keep_text_sim", 0.20)),
        )

    @staticmethod
    def _aggregate_nexus(
        detections: Sequence[Tuple[float, float, float]],
        frame_obj: int,
        box_conf_threshold: float = 0.30,
        text_sim_threshold: float = 0.20,
    ) -> float:
        """Port of OpenS2V ``eval/get_nexusscore.py`` filtering + aggregation.

        ``detections`` is a flat sequence of ``(bbox_conf, text_sim, image_sim)`` over
        all frames and detected boxes; ``frame_obj`` is the number of frames that
        contained at least one detection. A detection is kept only when

            bbox_conf > box_conf_threshold  (repo: 0.6 on YOLO-World scores)
            text_sim  > text_sim_threshold  (repo: 0.30 on GME text-image sim)
            image_sim != 0

        The kept image-image similarities are averaged and divided by ``frame_obj``
        (the reference's length normalization), matching upstream::

            nexus_score = mean(retrieval_score_list) / frame_obj

        Returns 0.0 when no detection passes the gates (repo behavior).
        """
        kept = [
            float(image_sim)
            for (bbox_conf, text_sim, image_sim) in detections
            if float(bbox_conf) > box_conf_threshold
            and float(text_sim) > text_sim_threshold
            and float(image_sim) != 0.0
        ]
        if not kept:
            return 0.0
        return float(np.mean(kept)) / max(int(frame_obj), 1)

    def _detect_subject(
        self, pil_frame: Image.Image, phrase: str
    ) -> List[Tuple[Image.Image, float]]:
        """Run GroundingDINO for the subject phrase; return (crop, confidence) list."""
        import torch

        text = phrase.strip().lower()
        if not text.endswith("."):
            text = text + " ."

        inputs = self._gdino_processor(
            images=pil_frame, text=text, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._gdino_model(**inputs)

        result = self._gdino_postprocess(outputs, inputs, pil_frame)
        boxes = result.get("boxes")
        scores = result.get("scores")
        if boxes is None or scores is None or len(boxes) == 0:
            return []

        width, height = pil_frame.size
        crops: List[Tuple[Image.Image, float]] = []
        for box, score in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = box
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(width, int(round(x2)))
            y2 = min(height, int(round(y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append((pil_frame.crop((x1, y1, x2, y2)), float(score)))
        return crops

    def _gdino_postprocess(
        self, outputs: Any, inputs: Any, pil_frame: Image.Image
    ) -> Dict[str, Any]:
        """Version-tolerant GroundingDINO post-processing (threshold kwarg renamed
        ``box_threshold`` -> ``threshold`` across transformers releases)."""
        import inspect

        fn = self._gdino_processor.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters
        target_sizes = [pil_frame.size[::-1]]  # (height, width)
        kwargs: Dict[str, Any] = {"target_sizes": target_sizes}
        box_thr = float(self.config.get("box_threshold", 0.30))
        if "threshold" in params:
            kwargs["threshold"] = box_thr
        elif "box_threshold" in params:
            kwargs["box_threshold"] = box_thr
        if "text_threshold" in params:
            kwargs["text_threshold"] = float(self.config.get("text_threshold", 0.25))
        # ``input_ids`` is a named parameter across transformers releases; pass it
        # by keyword when present so this is robust to argument reordering.
        if "input_ids" in params:
            kwargs["input_ids"] = inputs["input_ids"]
            return fn(outputs, **kwargs)[0]
        return fn(outputs, inputs["input_ids"], **kwargs)[0]

    # ------------------------------------------------------------------ #
    #  NaturalScore                                                       #
    # ------------------------------------------------------------------ #

    def _compute_natural(self, sample: Sample) -> Optional[float]:
        if self._natural_backend == "openai":
            return self._natural_openai(sample)
        return self._natural_vlm(sample)

    def _natural_frames_official(self, video_path: str, num_frames: int = 16):
        """Upstream ``extract_frames``: stride ``total//num_frames`` positions,
        long-side 512 resize, base64 JPEG."""
        import base64
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // num_frames, 1)
        frames_b64 = []
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if h >= w:
                    new_size = (int(w * 512 / h), 512)
                else:
                    new_size = (512, int(h * 512 / w))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode(".jpg", frame)
                frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
        cap.release()
        return frames_b64

    def _natural_openai(self, sample: Sample) -> Optional[float]:
        if not sample.is_video:
            return None
        frames_b64 = self._natural_frames_official(
            str(sample.path), int(self.config.get("natural_frames", 16))
        )
        if not frames_b64:
            return None

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
            for b64 in frames_b64
        ]
        content.append({"type": "text", "text": _NATURAL_PROMPT})

        scores: List[float] = []
        for _ in range(int(self.config.get("natural_runs", 3))):
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.config.get("openai_model", "gpt-4o-2024-11-20"),
                    stream=False,
                    messages=[{"role": "user", "content": content}],
                )
                text = response.choices[0].message.content.strip()
                m = re.search(r"[1-5]", text)
                if m:
                    scores.append(float(m.group()))
            except Exception as e:  # noqa: BLE001
                logger.info("OpenS2V: OpenAI judge call failed: %s", e)
        return float(np.mean(scores)) if scores else None

    def _natural_vlm(self, sample: Sample) -> Optional[float]:
        frames = sample_frames(
            sample.path,
            max_frames=int(self.config.get("vlm_max_frames", 4)),
            color="rgb",
        )
        if not frames:
            return None
        scores: List[int] = []
        for arr in frames:
            rating = self._vlm_rate(Image.fromarray(np.ascontiguousarray(arr)))
            if rating is not None:
                scores.append(rating)
        return float(np.mean(scores)) if scores else None

    def _vlm_rate(self, pil_frame: Image.Image) -> Optional[int]:
        try:
            import torch

            prompt = f"USER: <image>\n{_NATURAL_PROMPT}\nASSISTANT:"
            # LlavaProcessor.__call__ takes ``images`` as its first positional
            # argument, so pass text/images by keyword (matching the model card).
            inputs = self._vlm_processor(
                text=prompt, images=pil_frame, return_tensors="pt"
            ).to(self._device)
            with torch.inference_mode():
                out = self._vlm_model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get("vlm_max_new_tokens", 8)),
                )
                response = self._vlm_processor.decode(out[0], skip_special_tokens=True)
            response = response.split("ASSISTANT:")[-1]
            match = re.search(r"[1-5]", response)
            return int(match.group(0)) if match else None
        except Exception as e:  # noqa: BLE001
            logger.debug("OpenS2V VLM rating failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    #  Encoders                                                           #
    # ------------------------------------------------------------------ #

    def _clip_image_embeds(self, pils: List[Image.Image]) -> Any:
        import torch

        inputs = self._clip_processor(images=pils, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            feats = self._clip_model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    def _clip_text_embed(self, phrase: str) -> Any:
        import torch

        inputs = self._clip_processor(
            text=[phrase], return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.inference_mode():
            feats = self._clip_model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats.float(), dim=-1)[0]

    def _dino_image_embeds(self, pils: List[Image.Image]) -> Any:
        import torch

        batch = torch.stack([self._dino_transform(im) for im in pils]).to(self._device)
        with torch.inference_mode():
            feats = self._dino_model(batch)
            if isinstance(feats, dict) and "x" in feats:
                feats = feats["x"]
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    def _reference_embed(self, ref_path: Path, which: str) -> Optional[Any]:
        pils = self._reference_pils(ref_path)
        if not pils:
            return None
        import torch

        embeds = (
            self._clip_image_embeds(pils)
            if which == "clip"
            else self._dino_image_embeds(pils)
        )
        mean = embeds.mean(dim=0)
        return torch.nn.functional.normalize(mean, dim=-1)

    @staticmethod
    def _reference_pils(ref_path: Path) -> List[Image.Image]:
        pils: List[Image.Image] = []
        if ref_path.is_dir():
            for f in sorted(ref_path.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    im = load_pil_image(f)
                    if im is not None:
                        pils.append(im)
        elif ref_path.exists():
            im = load_pil_image(ref_path)
            if im is not None:
                pils.append(im)
        return pils

    # ------------------------------------------------------------------ #
    #  Input resolution                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_phrase(self, sample: Sample) -> Optional[str]:
        phrase = self.config.get("subject_prompt")
        if phrase:
            return str(phrase)
        if sample.caption and sample.caption.text:
            return sample.caption.text
        return None

    def _resolve_reference(self, sample: Sample) -> Optional[Path]:
        ref = getattr(sample, "reference_path", None)
        if ref:
            p = Path(ref)
            if p.exists():
                return p
        cfg_ref = self.config.get("reference_image")
        if cfg_ref:
            p = Path(cfg_ref)
            if p.exists():
                return p
        return None
