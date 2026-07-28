"""T2V-CompBench module — CVPR 2025.

Compositional metrics for text-to-video generation. The upstream T2V-CompBench
evaluators are model-specific per dimension:

  * MLLM (LLaVA / Grid-LLaVA) — consistent & dynamic attribute binding,
    action binding, object interactions.
  * Detection (GroundingDINO / GroundingSAM + Depth) — spatial relationships,
    generative numeracy.
  * Tracking (GroundingSAM + dense optical tracking) — motion binding.

This module keeps only the sub-metrics whose implementation matches the reference
evaluator class:

  * ``compbench_spatial`` / ``compbench_numeracy`` — computed with a real
    open-vocabulary object detector (+ Depth for behind/in-front), matching the
    upstream detection-based evaluators.
  * ``compbench_attribute`` / ``compbench_object_rel`` / ``compbench_action`` /
    ``compbench_scene`` — the upstream evaluator is an MLLM (LLaVA). A bare
    CLIP text-image cosine is a proxy, not that evaluator, so these are produced
    only when a real MLLM evaluator backend is available and are otherwise left
    unset (no CLIP-prompt fallback).

``compbench_overall`` is the weighted mean of the sub-metrics actually produced.
Video-only (requires caption).
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Number words for numeracy parsing
WORD_TO_NUM = {
    "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "couple": 2, "pair": 2,
    "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "several": -1, "many": -1, "multiple": -1, "few": -1,
}

# Spatial prepositions
SPATIAL_PREPS = [
    "above", "below", "left of", "right of", "behind", "in front of",
    "on top of", "under", "next to", "beside", "between", "inside",
    "outside", "near", "far from",
]

# Relationship words
RELATION_WORDS = [
    "holding", "carrying", "wearing", "riding", "sitting on",
    "standing on", "touching", "hugging", "pushing", "pulling",
    "chasing", "following", "leading", "watching",
]

# Action verbs (gerund form)
ACTION_VERBS = [
    "running", "walking", "jumping", "dancing", "swimming", "flying",
    "eating", "drinking", "sleeping", "talking", "laughing", "crying",
    "playing", "fighting", "climbing", "falling", "spinning", "turning",
    "waving", "pointing", "throwing", "catching", "kicking",
]


class T2VCompBenchModule(PipelineModule):
    name = "t2v_compbench"
    description = "T2V-CompBench compositional metrics (detection spatial/numeracy; MLLM for the rest)"
    default_config = {
        "subsample": 8,
        "enable_attribute": True,
        "enable_object_rel": True,
        "enable_action": True,
        "enable_spatial": True,
        "enable_numeracy": True,
        "enable_scene": True,
        "weights": [1, 1, 1, 1, 1, 1],
    }
    metric_groups = {
        "compbench_action": "alignment",
        "compbench_attribute": "alignment",
        "compbench_numeracy": "alignment",
        "compbench_object_rel": "alignment",
        "compbench_overall": "alignment",
        "compbench_scene": "alignment",
        "compbench_spatial": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._detector_available = False
        self._mllm_available = False
        self._yolo = None
        self._depth = None
        self._mllm = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except ImportError:
            self._device = "cpu"

        detector_ok = self._try_load_detector()
        depth_ok = self._try_load_depth() if detector_ok else False
        mllm_ok = self._try_load_mllm()

        self._detector_available = detector_ok
        self._mllm_available = mllm_ok

        if detector_ok or mllm_ok:
            self._ml_available = True
            parts = []
            if detector_ok:
                parts.append("detector_depth" if depth_ok else "detector")
            if mllm_ok:
                parts.append("mllm")
            self._backend = "+".join(parts)
            logger.info("T2VCompBench using %s backend", self._backend)
        else:
            self._backend = "unavailable"
            logger.info(
                "T2VCompBench unavailable: neither an open-vocabulary detector "
                "(spatial/numeracy) nor an MLLM evaluator (attribute/object_rel/"
                "action/scene) could be loaded; no compbench_* metrics will be "
                "populated."
            )

    def _try_load_detector(self) -> bool:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8s-world.pt")
            return True
        except Exception as e:
            logger.info("Open-vocabulary detector unavailable: %s", e)
            return False

    def _try_load_depth(self) -> bool:
        try:
            from transformers import pipeline
            self._depth = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self._device == "cuda" else -1,
            )
            return True
        except Exception as e:
            logger.info("Depth Anything unavailable: %s", e)
            return False

    def _try_load_mllm(self) -> bool:
        """Load the upstream MLLM (LLaVA/Grid-LLaVA) T2V-CompBench evaluator.

        The attribute / object-interaction / action / scene dimensions are
        MLLM-scored in the upstream benchmark. There is no installable backend
        for that evaluator here, so this returns False and those dimensions are
        left unset rather than substituting a CLIP-prompt proxy.
        """
        try:
            import t2v_compbench_eval  # type: ignore  # upstream MLLM evaluator
            self._mllm = t2v_compbench_eval
            return True
        except ImportError:
            return False
        except Exception as e:  # pragma: no cover - defensive
            logger.info("T2V-CompBench MLLM evaluator unavailable: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample
        if not sample.caption or not sample.caption.text:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            caption = sample.caption.text
            frames = self._extract_frames(sample)
            if len(frames) < 2:
                return sample

            scores: Dict[str, float] = {}
            if self._detector_available:
                scores.update(self._compute_detection(frames, caption))
            if self._mllm_available:
                scores.update(self._compute_mllm(frames, caption))

            qm = sample.quality_metrics
            if self.config.get("enable_attribute", True) and "attribute" in scores:
                qm.compbench_attribute = scores["attribute"]
            if self.config.get("enable_object_rel", True) and "object_rel" in scores:
                qm.compbench_object_rel = scores["object_rel"]
            if self.config.get("enable_action", True) and "action" in scores:
                qm.compbench_action = scores["action"]
            if self.config.get("enable_spatial", True) and "spatial" in scores:
                qm.compbench_spatial = scores["spatial"]
            if self.config.get("enable_numeracy", True) and "numeracy" in scores:
                qm.compbench_numeracy = scores["numeracy"]
            if self.config.get("enable_scene", True) and "scene" in scores:
                qm.compbench_scene = scores["scene"]

            weights = self.config.get("weights", [1, 1, 1, 1, 1, 1])
            sub_keys = ["attribute", "object_rel", "action", "spatial", "numeracy", "scene"]
            valid_scores = []
            valid_weights = []
            for i, key in enumerate(sub_keys):
                if key in scores and scores[key] is not None:
                    valid_scores.append(scores[key])
                    valid_weights.append(weights[i] if i < len(weights) else 1)

            if valid_scores:
                w_sum = sum(valid_weights)
                overall = sum(s * w for s, w in zip(valid_scores, valid_weights)) / (w_sum + 1e-8)
                qm.compbench_overall = float(np.clip(overall, 0.0, 1.0))

        except Exception as e:
            logger.warning("T2VCompBench processing failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Frame extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_frames(self, sample: Sample) -> list:
        num_frames = self.config.get("subsample", 8)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return []
        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------ #
    # Text parsing utilities                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_attributes(caption: str) -> List[Tuple[str, str]]:
        """Parse 'adjective noun' pairs from caption."""
        pattern = re.compile(
            r'\b(red|blue|green|yellow|black|white|large|small|big|tiny|tall|short|'
            r'old|young|new|bright|dark|round|square|long|thin|thick|heavy|light|'
            r'beautiful|ugly|colorful|shiny|smooth|rough|soft|hard|fast|slow|'
            r'wooden|metal|glass|golden|silver|furry|striped|spotted)\s+'
            r'(\w+)\b', re.IGNORECASE
        )
        return pattern.findall(caption)

    @staticmethod
    def _parse_spatial(caption: str) -> List[Tuple[str, str, str]]:
        """Parse 'X spatial_prep Y' patterns."""
        results = []
        caption_lower = caption.lower()
        for prep in SPATIAL_PREPS:
            pattern = re.compile(r'(\w+)\s+' + re.escape(prep) + r'\s+(?:the\s+)?(\w+)', re.IGNORECASE)
            for m in pattern.finditer(caption_lower):
                results.append((m.group(1), prep, m.group(2)))
        return results

    @staticmethod
    def _parse_count(caption: str) -> List[Tuple[int, str]]:
        """Parse 'number noun' patterns."""
        results = []
        number_words = set(WORD_TO_NUM.keys())
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in WORD_TO_NUM) + r'|\d+)\s+(\w+(?:\s+\w+)?)\b',
            re.IGNORECASE
        )
        for m in pattern.finditer(caption):
            word = m.group(1).lower()
            rest = m.group(2).strip()
            parts = rest.split()
            noun = parts[-1]
            if noun.lower() in number_words:
                continue
            if word in WORD_TO_NUM:
                num = WORD_TO_NUM[word]
            else:
                try:
                    num = int(word)
                except ValueError:
                    continue
            if num >= 0:
                results.append((num, noun))
        return results

    @staticmethod
    def _parse_actions(caption: str) -> List[Tuple[str, str]]:
        """Parse 'noun verb-ing' patterns."""
        results = []
        caption_lower = caption.lower()
        for verb in ACTION_VERBS:
            pattern = re.compile(r'(\w+)\s+' + re.escape(verb), re.IGNORECASE)
            for m in pattern.finditer(caption_lower):
                results.append((m.group(1), verb))
        return results

    @staticmethod
    def _parse_relations(caption: str) -> List[Tuple[str, str, str]]:
        """Parse 'X relation Y' patterns."""
        results = []
        caption_lower = caption.lower()
        for rel in RELATION_WORDS:
            pattern = re.compile(r'(\w+)\s+' + re.escape(rel) + r'\s+(?:a\s+|the\s+)?(\w+)', re.IGNORECASE)
            for m in pattern.finditer(caption_lower):
                results.append((m.group(1), rel, m.group(2)))
        return results

    # ------------------------------------------------------------------ #
    # Detection-based evaluators (spatial, numeracy) — matches upstream   #
    # GroundingDINO/GroundingSAM detector class.                          #
    # ------------------------------------------------------------------ #

    def _detect_objects(self, frame: np.ndarray, classes: List[str]) -> list:
        """Run open-vocabulary detection for specific classes."""
        self._yolo.set_classes(classes)
        results = self._yolo(frame, verbose=False)
        detections = []
        if results and len(results) > 0:
            r = results[0]
            for box, cls_id, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                          r.boxes.cls.cpu().numpy().astype(int),
                                          r.boxes.conf.cpu().numpy()):
                detections.append({
                    "class": classes[cls_id] if cls_id < len(classes) else "unknown",
                    "box": box.tolist(),
                    "conf": float(conf),
                })
        return detections

    def _compute_detection(self, frames: list, caption: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # Spatial relationship — detection + geometric/depth rule
        spatials = self._parse_spatial(caption)
        if spatials:
            spatial_scores = []
            for subj, prep, obj in spatials:
                for frame in frames[::max(1, len(frames) // 4)]:
                    dets_subj = self._detect_objects(frame, [subj])
                    dets_obj = self._detect_objects(frame, [obj])
                    if dets_subj and dets_obj:
                        s_box = dets_subj[0]["box"]
                        o_box = dets_obj[0]["box"]
                        spatial_ok = self._verify_spatial(s_box, o_box, prep, frame)
                        spatial_scores.append(1.0 if spatial_ok else 0.0)
                    else:
                        spatial_scores.append(0.0)
            if spatial_scores:
                scores["spatial"] = float(np.mean(spatial_scores))

        # Generative numeracy — detection + count comparison
        counts = self._parse_count(caption)
        if counts:
            count_scores = []
            for expected_num, noun in counts:
                frame_counts = []
                for frame in frames[::max(1, len(frames) // 4)]:
                    dets = self._detect_objects(frame, [noun.rstrip("s")])
                    frame_counts.append(len(dets))
                avg_count = np.mean(frame_counts) if frame_counts else 0
                if expected_num == 0:
                    count_scores.append(1.0 if avg_count == 0 else 0.0)
                else:
                    count_scores.append(max(0.0, 1.0 - abs(avg_count - expected_num) / expected_num))
            if count_scores:
                scores["numeracy"] = float(np.mean(count_scores))

        return scores

    def _verify_spatial(self, s_box: list, o_box: list, prep: str, frame: np.ndarray) -> bool:
        """Verify spatial relationship between two bounding boxes."""
        s_cx = (s_box[0] + s_box[2]) / 2
        s_cy = (s_box[1] + s_box[3]) / 2
        o_cx = (o_box[0] + o_box[2]) / 2
        o_cy = (o_box[1] + o_box[3]) / 2

        if prep in ("above", "on top of"):
            return s_cy < o_cy
        elif prep in ("below", "under"):
            return s_cy > o_cy
        elif prep == "left of":
            return s_cx < o_cx
        elif prep == "right of":
            return s_cx > o_cx
        elif prep in ("next to", "beside", "near"):
            dist = np.sqrt((s_cx - o_cx) ** 2 + (s_cy - o_cy) ** 2)
            diag = np.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
            return dist < diag * 0.3
        elif prep in ("behind", "in front of"):
            if self._depth is not None:
                try:
                    from PIL import Image
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    depth_result = self._depth(pil_img)
                    depth_map = np.array(depth_result["depth"])
                    s_depth = depth_map[int(s_cy):int(s_box[3]), int(s_box[0]):int(s_box[2])].mean()
                    o_depth = depth_map[int(o_cy):int(o_box[3]), int(o_box[0]):int(o_box[2])].mean()
                    if prep == "behind":
                        return s_depth > o_depth
                    else:
                        return s_depth < o_depth
                except Exception:
                    pass
            return True  # Can't verify without depth
        return True

    # ------------------------------------------------------------------ #
    # MLLM-based evaluators (attribute, object_rel, action, scene)        #
    # ------------------------------------------------------------------ #

    def _compute_mllm(self, frames: list, caption: str) -> Dict[str, float]:
        """Score the MLLM dimensions with the upstream LLaVA-based evaluator.

        Only reached when a real MLLM evaluator backend is loaded; otherwise the
        attribute / object_rel / action / scene dimensions are left unset.
        """
        scores: Dict[str, float] = {}
        evaluate = getattr(self._mllm, "evaluate", None)
        if evaluate is None:
            return scores
        try:
            result = evaluate(frames, caption)
        except Exception as e:
            logger.warning("T2VCompBench MLLM evaluation failed: %s", e)
            return scores
        if not isinstance(result, dict):
            return scores
        for key in ("attribute", "object_rel", "action", "scene"):
            if result.get(key) is not None:
                scores[key] = float(result[key])
        return scores
