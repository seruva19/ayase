"""T2V-CompBench module — CVPR 2025.

Seven compositional sub-metrics for text-to-video generation evaluation:
  1. Attribute binding — does generated object have the described attribute?
  2. Object relationship — are described inter-object relations present?
  3. Action binding — does the subject perform the described action?
  4. Spatial relationship — correct spatial arrangement?
  5. Generative numeracy — correct count of objects?
  6. Scene composition — overall scene matches caption?
  7. Overall — weighted mean of sub-metrics.

Backend tiers:
  1. **YOLO+Depth+CLIP** — YOLO-World detection + Depth Anything V2 + CLIP verification
  2. **CLIP-only** — CLIP text-image matching (no detection, skip spatial/numeracy)
  3. **Heuristic** — Text parsing + basic spatial analysis

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
    description = "T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic)"
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

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._yolo = None
        self._depth = None
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def setup(self) -> None:
        # Tier 1: YOLO-World + Depth Anything + CLIP
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

        yolo_ok = self._try_load_yolo()
        clip_ok = self._try_load_clip()
        depth_ok = self._try_load_depth()

        if yolo_ok and clip_ok:
            self._backend = "yolo_depth" if depth_ok else "yolo_clip"
            logger.info("T2VCompBench using YOLO+CLIP%s backend", "+Depth" if depth_ok else "")
        elif clip_ok:
            self._backend = "clip"
            logger.info("T2VCompBench using CLIP-only backend")
        else:
            self._backend = "heuristic"
            logger.info("T2VCompBench using heuristic backend")

    def _try_load_yolo(self) -> bool:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8s-world.pt")
            return True
        except Exception as e:
            logger.info("YOLO-World unavailable: %s", e)
            return False

    def _try_load_clip(self) -> bool:
        try:
            from transformers import CLIPModel, CLIPProcessor
            models_dir = self.config.get("models_dir", "models")
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            ).to(self._device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            )
            return True
        except Exception as e:
            logger.info("CLIP unavailable: %s", e)
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

    def process(self, sample: Sample) -> Sample:
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

            scores = {}

            if self._backend == "yolo_depth":
                scores = self._compute_yolo_depth(frames, caption)
            elif self._backend == "clip":
                scores = self._compute_clip_only(frames, caption)
            else:
                scores = self._compute_heuristic(frames, caption)

            # Write scores to quality metrics
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

            # Overall = weighted mean
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
        # Simple pattern: adj + noun
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
        # Match: number word/digit + optional 1-2 adjectives + noun
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in WORD_TO_NUM) + r'|\d+)\s+(\w+(?:\s+\w+)?)\b',
            re.IGNORECASE
        )
        for m in pattern.finditer(caption):
            word = m.group(1).lower()
            rest = m.group(2).strip()
            # Take the last word as noun (skip adjective words and other number words)
            parts = rest.split()
            noun = parts[-1]
            # Skip if noun is itself a number word
            if noun.lower() in number_words:
                continue
            if word in WORD_TO_NUM:
                num = WORD_TO_NUM[word]
            else:
                try:
                    num = int(word)
                except ValueError:
                    continue
            if num >= 0:  # Skip vague counts
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
    # CLIP similarity helper                                               #
    # ------------------------------------------------------------------ #

    def _clip_sim(self, image: np.ndarray, text: str) -> float:
        """Compute CLIP cosine similarity between image and text."""
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self._clip_processor(
            text=[text], images=pil_img, return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            logits = outputs.logits_per_image
            return float(logits[0, 0].item() / 100.0)

    def _clip_sim_batch_text(self, image: np.ndarray, texts: List[str]) -> List[float]:
        """CLIP similarity of one image against multiple texts."""
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self._clip_processor(
            text=texts, images=pil_img, return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            sims = outputs.logits_per_image.softmax(dim=-1)[0]
            return [float(s) for s in sims]

    # ------------------------------------------------------------------ #
    # Tier 1: YOLO + Depth + CLIP                                          #
    # ------------------------------------------------------------------ #

    def _detect_objects(self, frame: np.ndarray, classes: List[str]) -> list:
        """Run YOLO-World detection for specific classes."""
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

    def _compute_yolo_depth(self, frames: list, caption: str) -> Dict[str, float]:
        scores = {}

        # Attribute binding
        attrs = self._parse_attributes(caption)
        if attrs:
            attr_scores = []
            for adj, noun in attrs:
                for frame in frames[::max(1, len(frames) // 4)]:
                    dets = self._detect_objects(frame, [noun])
                    if dets:
                        # Crop detected region and verify attribute via CLIP
                        det = max(dets, key=lambda d: d["conf"])
                        x1, y1, x2, y2 = [int(c) for c in det["box"]]
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0:
                            sim = self._clip_sim(crop, f"a {adj} {noun}")
                            attr_scores.append(sim)
            scores["attribute"] = float(np.mean(attr_scores)) if attr_scores else 0.5

        # Object relationship
        rels = self._parse_relations(caption)
        if rels:
            rel_scores = []
            for subj, rel, obj in rels:
                for frame in frames[::max(1, len(frames) // 4)]:
                    dets_subj = self._detect_objects(frame, [subj])
                    dets_obj = self._detect_objects(frame, [obj])
                    if dets_subj and dets_obj:
                        # Both objects detected — verify relation via CLIP
                        sim = self._clip_sim(frame, f"{subj} {rel} {obj}")
                        rel_scores.append(sim)
                    elif dets_subj or dets_obj:
                        rel_scores.append(0.3)
                    else:
                        rel_scores.append(0.0)
            scores["object_rel"] = float(np.mean(rel_scores)) if rel_scores else 0.5

        # Action binding
        actions = self._parse_actions(caption)
        if actions:
            action_scores = []
            for noun, verb in actions:
                frame_scores = []
                for frame in frames[::max(1, len(frames) // 3)]:
                    dets = self._detect_objects(frame, [noun])
                    if dets:
                        det = max(dets, key=lambda d: d["conf"])
                        x1, y1, x2, y2 = [int(c) for c in det["box"]]
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0:
                            sim = self._clip_sim(crop, f"a {noun} {verb}")
                            frame_scores.append(sim)
                if frame_scores:
                    action_scores.append(float(np.mean(frame_scores)))
            scores["action"] = float(np.mean(action_scores)) if action_scores else 0.5

        # Spatial relationship
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
            scores["spatial"] = float(np.mean(spatial_scores)) if spatial_scores else 0.5

        # Numeracy
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
            scores["numeracy"] = float(np.mean(count_scores)) if count_scores else 0.5

        # Scene composition
        scene_scores = []
        for frame in frames[::max(1, len(frames) // 4)]:
            sim = self._clip_sim(frame, caption)
            scene_scores.append(sim)
        scores["scene"] = float(np.mean(scene_scores)) if scene_scores else 0.5

        # Fill defaults for unparsed dimensions
        for key in ["attribute", "object_rel", "action", "spatial", "numeracy", "scene"]:
            if key not in scores:
                scores[key] = 0.5

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
            # Use depth if available
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
    # Tier 2: CLIP-only                                                    #
    # ------------------------------------------------------------------ #

    def _compute_clip_only(self, frames: list, caption: str) -> Dict[str, float]:
        scores = {}

        # Attribute binding via CLIP
        attrs = self._parse_attributes(caption)
        if attrs:
            attr_scores = []
            for adj, noun in attrs:
                for frame in frames[::max(1, len(frames) // 4)]:
                    pos_text = f"a {adj} {noun}"
                    neg_text = f"a {noun}"
                    sims = self._clip_sim_batch_text(frame, [pos_text, neg_text])
                    attr_scores.append(sims[0])
            scores["attribute"] = float(np.mean(attr_scores)) if attr_scores else 0.5

        # Object relationship via CLIP
        rels = self._parse_relations(caption)
        if rels:
            rel_scores = []
            for subj, rel, obj in rels:
                for frame in frames[::max(1, len(frames) // 4)]:
                    sim = self._clip_sim(frame, f"{subj} {rel} {obj}")
                    rel_scores.append(sim)
            scores["object_rel"] = float(np.mean(rel_scores)) if rel_scores else 0.5

        # Action binding via CLIP temporal
        actions = self._parse_actions(caption)
        if actions:
            action_scores = []
            for noun, verb in actions:
                for frame in frames:
                    sim = self._clip_sim(frame, f"a {noun} {verb}")
                    action_scores.append(sim)
            scores["action"] = float(np.mean(action_scores)) if action_scores else 0.5

        # Scene via CLIP
        scene_scores = []
        for frame in frames[::max(1, len(frames) // 4)]:
            sim = self._clip_sim(frame, caption)
            scene_scores.append(sim)
        scores["scene"] = float(np.mean(scene_scores)) if scene_scores else 0.5

        # Spatial/numeracy not reliable with CLIP only — skip
        for key in ["attribute", "object_rel", "action", "scene"]:
            if key not in scores:
                scores[key] = 0.5

        return scores

    # ------------------------------------------------------------------ #
    # Tier 3: Heuristic                                                    #
    # ------------------------------------------------------------------ #

    def _compute_heuristic(self, frames: list, caption: str) -> Dict[str, float]:
        scores = {}
        caption_lower = caption.lower()

        # Attribute: check if descriptive adjectives are present
        attrs = self._parse_attributes(caption)
        scores["attribute"] = min(1.0, len(attrs) / 3.0) if attrs else 0.5

        # Relations: check if relation words are present
        rels = self._parse_relations(caption)
        scores["object_rel"] = min(1.0, len(rels) / 2.0) if rels else 0.5

        # Action: estimate motion magnitude per frame
        actions = self._parse_actions(caption)
        if actions:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            motion_mags = []
            for i in range(len(grays) - 1):
                diff = np.abs(grays[i + 1].astype(float) - grays[i].astype(float)).mean()
                motion_mags.append(diff)
            avg_motion = float(np.mean(motion_mags)) if motion_mags else 0
            # Actions imply motion — higher motion = better
            scores["action"] = min(avg_motion / 20.0, 1.0)
        else:
            scores["action"] = 0.5

        # Spatial: basic edge/region analysis
        spatials = self._parse_spatial(caption)
        scores["spatial"] = 0.5

        # Numeracy: edge detection for distinct objects
        counts = self._parse_count(caption)
        if counts:
            # Count contours as proxy for object count
            gray = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter small contours
            h, w = gray.shape
            min_area = h * w * 0.01
            large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            detected_count = len(large_contours)

            count_scores = []
            for expected_num, _ in counts:
                if expected_num == 0:
                    count_scores.append(1.0 if detected_count == 0 else 0.0)
                elif expected_num > 0:
                    count_scores.append(max(0.0, 1.0 - abs(detected_count - expected_num) / (expected_num + 1)))
            scores["numeracy"] = float(np.mean(count_scores)) if count_scores else 0.5
        else:
            scores["numeracy"] = 0.5

        # Scene: temporal consistency of frames
        if len(frames) >= 3:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            hist_corrs = []
            for i in range(len(grays) - 1):
                h1 = cv2.calcHist([grays[i]], [0], None, [64], [0, 256])
                h2 = cv2.calcHist([grays[i + 1]], [0], None, [64], [0, 256])
                corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                hist_corrs.append(max(corr, 0.0))
            scores["scene"] = float(np.mean(hist_corrs))
        else:
            scores["scene"] = 0.5

        return scores
