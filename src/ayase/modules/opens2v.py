"""OpenS2V-Eval subject-consistency metrics (NexusScore + NaturalScore).

Implements the two subject-driven video-generation metrics from OpenS2V-Nexus
(PKU-YuanGroup, arXiv:2505.20292) — the reference benchmark for subject-driven /
personalized ("subject-to-video", S2V) generation:

* **NexusScore** ``opens2v_nexus_score`` — subject consistency. An open-vocabulary
  detector localizes the *subject phrase* in every sampled frame, each detected
  crop is embedded and compared (cosine) against the reference subject image, and
  the per-detection similarities are filtered and aggregated. The text-relevance
  gate makes the score robust to multi-subject frames: crops that do not match the
  subject phrase (the *wrong* subject) are dropped before aggregation.
* **NaturalScore** ``opens2v_natural_score`` — naturalness. A vision-language
  model rates each frame on OpenS2V's 1-5 anti-"copy-paste" rubric (common sense,
  physical plausibility, visual naturalness, AI-generation artifacts).

Fidelity vs the reference implementation (``eval/get_nexusscore.py`` /
``eval/get_naturalscore.py`` in the OpenS2V-Nexus repo):

    Component            | OpenS2V reference        | This module (practical, real-or-none)
    ---------------------|--------------------------|--------------------------------------
    detector             | YOLO-World v2 image-     | GroundingDINO (HF
                         | prompt adapter (MMEngine)| ``IDEA-Research/grounding-dino-*``),
                         |                          | text-prompted with the subject phrase
    crop encoder         | GME (Qwen2-VL-7B) image  | CLIP-I (default) or DINOv2 — config
                         | & text embeddings        | ``encoder``; CLIP also drives the
                         |                          | text-relevance gate
    aggregation          | ``mean(kept)/frame_obj`` | identical (see ``_aggregate_nexus``)
    keep gates           | bbox>0.6, text>0.30,     | configurable ``keep_box_conf`` /
                         | image!=0                 | ``keep_text_sim`` (defaults recalibrated
                         |                          | for the GDINO/CLIP confidence scale)
    NaturalScore judge   | GPT-4o (whole video, x3) | local VLM (LLaVA, per frame, averaged)
                         |                          | — same 1-5 rubric, no external API

The reference substitutions (GroundingDINO text-prompted detector; CLIP/DINO
crop encoder; local VLM judge) are deliberate: YOLO-World's MMEngine
image-prompt-adapter stack and GME-Qwen2-VL-7B / GPT-4o are impractical hard
dependencies, and a text-prompted detector matches "detect the subject phrase per
frame" directly. The filtering + aggregation math is ported exactly.

STRICT real-or-none policy: there is **no** whole-frame-similarity fallback (that
is what ``clip_image_similarity`` / ``i2v_similarity`` already do). If the
detector or crop encoder are unavailable, ``opens2v_nexus_score`` stays ``None``;
if the VLM is unavailable, ``opens2v_natural_score`` stays ``None``. ``self._backend``
records what actually ran (e.g. ``"groundingdino+clip"``, ``"groundingdino+clip+llava"``,
or ``"unavailable"``).

Reference subject image (precedence): ``sample.reference_path`` (an image file, or
a directory whose images are averaged), else config ``reference_image``.
Subject phrase (precedence): config ``subject_prompt``, else ``sample.caption.text``.
When neither a reference nor a phrase can be resolved, NexusScore is left ``None``.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ayase.image import IMAGE_EXTENSIONS, load_pil_image, sample_frames
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# DINOv2 backbone weights (mirror, consistent with dino_face_identity/i2v_similarity).
_DINOV2_MIRROR_BASE = "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/"
_DINOV2_WEIGHTS = {
    "dinov2_vitb14": "dino_face_identity/dinov2_vitb14_pretrain.pth",
}

# OpenS2V NaturalScore rubric (ported from eval/get_naturalscore.py — GPT-4o prompt).
_NATURAL_RUBRIC = (
    "You are evaluating whether a single video frame looks natural and free of "
    "AI-generation artifacts. Assess: (1) common-sense consistency, (2) physical "
    "plausibility (lighting, shadows, motion, reflections), (3) visual naturalness "
    "(textures, proportions, details), and (4) AI-generation artifacts (blurring, "
    "morphing, glitches, distortions). If people are present, pay special attention "
    "to faces, anatomical correctness, and body proportions. Rate the naturalness on "
    "an integer scale from 1 to 5, where: 1 = clear and frequent artifacts, distorted "
    "shapes, implausible physics; 2 = noticeable AI-generation cues, inconsistent "
    "anatomy, fluctuating textures; 3 = mixed indicators, subtle flaws or small "
    "anomalies; 4 = mostly natural, only minor and rare irregularities; 5 = fully "
    "consistent with real-world physics, no visible artifacts. Answer with a single "
    "digit from 1 to 5 and nothing else."
)


class OpenS2VModule(PipelineModule):
    name = "opens2v"
    description = (
        "OpenS2V-Eval subject-consistency metrics: NexusScore (GroundingDINO subject "
        "crops vs reference subject image) and NaturalScore (VLM naturalness judge)"
    )
    default_config = {
        "device": "auto",
        "max_frames": 16,
        # GroundingDINO (open-vocabulary, text-prompted) detector
        "detector_model": "IDEA-Research/grounding-dino-tiny",
        "box_threshold": 0.30,   # GDINO detection score threshold (post-process)
        "text_threshold": 0.25,  # GDINO token-grounding threshold (post-process)
        # NexusScore aggregation gates (OpenS2V get_nexusscore.py: 0.6 / 0.30 for
        # YOLO-World + GME; recalibrated here for the GDINO + CLIP confidence scale).
        "keep_box_conf": 0.30,
        "keep_text_sim": 0.20,
        # Subject-crop feature encoder for crop<->reference similarity.
        "encoder": "clip",       # "clip" (CLIP-I) | "dino" (DINOv2)
        "clip_model": "openai/clip-vit-base-patch32",
        "dino_model": "dinov2_vitb14",
        # NaturalScore VLM judge
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
            "reference subject image (0-1 range before frame normalization; higher=better)"
        ),
        "opens2v_natural_score": (
            "OpenS2V NaturalScore — VLM naturalness rating on the 1-5 anti-copy-paste "
            "rubric (higher=better)"
        ),
    }

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

        # Availability flags
        self._gdino_ok = False
        self._clip_ok = False
        self._dino_ok = False
        self._vlm_ok = False
        self._nexus_available = False
        self._natural_available = False
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

        self._load_detector()
        self._load_clip()
        if self._encoder == "dino":
            self._load_dino()
        self._load_vlm()

        # NexusScore needs the detector, the CLIP text gate, and the chosen crop encoder.
        self._nexus_available = (
            self._gdino_ok
            and self._clip_ok
            and (self._encoder != "dino" or self._dino_ok)
        )
        self._natural_available = self._vlm_ok

        parts: List[str] = []
        if self._gdino_ok:
            parts.append("groundingdino")
        if self._encoder == "dino" and self._dino_ok:
            parts.append("dino")
        if self._clip_ok:
            parts.append("clip")
        if self._vlm_ok:
            parts.append(self._vlm_tag)

        if (self._nexus_available or self._natural_available) and parts:
            self._backend = "+".join(parts)
        else:
            self._backend = "unavailable"

        logger.info(
            "OpenS2V ready: backend=%s nexus=%s natural=%s",
            self._backend,
            self._nexus_available,
            self._natural_available,
        )

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
                from ayase.config import download_model_file

                ckpt = download_model_file(rel, _DINOV2_MIRROR_BASE + rel, models_dir)
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
            from transformers import (
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
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
                        LlavaNextForConditionalGeneration,
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
                processor = LlavaNextProcessor.from_pretrained(name, cache_dir=models_dir)
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
            frames = sample_frames(
                sample.path, max_frames=int(self.config.get("max_frames", 16)), color="rgb"
            )
            if not frames:
                return sample

            nexus: Optional[float] = None
            if self._nexus_available:
                nexus = self._compute_nexus(sample, frames)

            natural: Optional[float] = None
            if self._natural_available:
                natural = self._compute_natural(frames)

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

    def _compute_nexus(self, sample: Sample, frames: List[np.ndarray]) -> Optional[float]:
        phrase = self._resolve_phrase(sample)
        ref_path = self._resolve_reference(sample)
        if not phrase or ref_path is None:
            # No subject phrase or reference subject image => cannot run (strict None).
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
            box_conf_threshold=float(self.config.get("keep_box_conf", 0.30)),
            text_sim_threshold=float(self.config.get("keep_text_sim", 0.20)),
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
        return fn(outputs, inputs["input_ids"], **kwargs)[0]

    # ------------------------------------------------------------------ #
    #  NaturalScore                                                       #
    # ------------------------------------------------------------------ #

    def _compute_natural(self, frames: List[np.ndarray]) -> Optional[float]:
        max_frames = int(self.config.get("vlm_max_frames", 4))
        step = max(1, len(frames) // max_frames) if max_frames > 0 else 1
        scores: List[int] = []
        for frame in frames[::step][:max_frames]:
            pil = Image.fromarray(np.ascontiguousarray(frame))
            rating = self._vlm_rate(pil)
            if rating is not None:
                scores.append(rating)
        if not scores:
            return None
        return float(np.mean(scores))

    def _vlm_rate(self, pil_frame: Image.Image) -> Optional[int]:
        try:
            import torch

            prompt = f"USER: <image>\n{_NATURAL_RUBRIC}\nASSISTANT:"
            inputs = self._vlm_processor(
                prompt, images=pil_frame, return_tensors="pt"
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
    #  Reference / subject-phrase resolution                             #
    # ------------------------------------------------------------------ #

    def _resolve_phrase(self, sample: Sample) -> Optional[str]:
        prompt = self.config.get("subject_prompt")
        if prompt:
            return str(prompt)
        caption = getattr(sample, "caption", None)
        if caption is not None and getattr(caption, "text", None):
            return str(caption.text)
        return None

    def _resolve_reference(self, sample: Sample) -> Optional[Path]:
        ref = getattr(sample, "reference_path", None)
        if ref:
            rp = Path(ref)
            if rp.exists():
                return rp
        cfg_ref = self.config.get("reference_image")
        if cfg_ref:
            rp = Path(str(cfg_ref))
            if rp.exists():
                return rp
        return None
