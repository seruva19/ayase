import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from PIL import Image

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# VLM Classification Presets
# ------------------------------------------------------------------
VLM_PRESETS: Dict[str, List[str]] = {
    "shot_scale": [
        "extreme_close_up", "close_up", "medium", "full", "wide", "extreme_wide",
    ],
    "time_of_day": [
        "dawn", "morning", "midday", "afternoon", "dusk", "night",
    ],
    "clothing_style": [
        "casual", "formal", "sportswear", "costume", "uniform", "none_visible",
    ],
    "mood": [
        "joyful", "calm", "tense", "sad", "energetic", "neutral",
    ],
    "expression": [
        "happy", "sad", "angry", "surprised", "neutral", "fearful", "no_face",
    ],
}


class VLMJudgeModule(PipelineModule):
    """
    Advanced semantic reasoning module using Vision-Language Models (VLM).
    Detects hallucinations in captions and verifies complex spatial/logical relationships.
    Supports three modes: 'verify', 'traits', and 'presets'.
    """
    name = "vlm_judge"
    description = "Advanced semantic verification using VLM (e.g. LLaVA)"
    default_config = {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "max_new_tokens": 256,
        "mode": "verify",  # 'verify' | 'traits' | 'presets'
        "verification_prompt": "Does the following text accurately describe the image? Answer only with 'Yes' or 'No' and explain why briefly. Text: '{caption}'",
        "traits_prompt": "Analyze this video and identify its distinctive visual style characteristics. Format your response as a JSON object where keys are trait names and values are scores (0-1). Example: {'warm_tones': 0.8, 'cinematic': 0.9}",
        "presets": [],  # list of preset names to evaluate (e.g. ["shot_scale", "mood"])
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "llava-hf/llava-1.5-7b-hf")
        self.max_new_tokens = self.config.get("max_new_tokens", 128)
        self.verification_prompt = self.config.get("verification_prompt")

        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading VLM Judge ({self.model_name}) on {self._device}...")

            models_dir = self.config.get("models_dir", "models")

            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                cache_dir=models_dir,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._model.eval()
            self._processor = LlavaNextProcessor.from_pretrained(self.model_name, cache_dir=models_dir)

            self._ml_available = True

        except ImportError:
            logger.warning("Transformers / LLaVA dependencies not met. VLM Judge disabled.")
        except Exception as e:
            logger.warning(f"Failed to setup VLM Judge: {e}")

    def process(self, sample: Sample) -> Sample:
        mode = self.config.get("mode", "verify")

        # Presets mode doesn't require captions — only images
        if mode == "presets":
            return self._process_presets(sample)

        # verify / traits modes require a caption
        if not sample.caption:
            return sample

        if not self._ml_available:
            return sample

        try:
            import torch
            image = self._load_image(sample)
            if image is None:
                return sample

            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if mode == "traits":
                prompt = f"USER: <image>\n{self.config.get('traits_prompt')}\nASSISTANT:"
            else:
                prompt = f"USER: <image>\n{self.verification_prompt.format(caption=sample.caption.text if sample.caption else '')}\nASSISTANT:"

            inputs = self._processor(prompt, images=pil_image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                response = self._processor.decode(output[0], skip_special_tokens=True)

            response_clean = response.split("ASSISTANT:")[-1].strip()

            if mode == "traits":
                import json
                import re
                try:
                    json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                    if json_match:
                        traits = json.loads(json_match.group(0))
                        sample.detections.append({"type": "style_traits", "data": traits})
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                message="VLM Style Analysis completed.",
                                details={"traits": traits}
                            )
                        )
                except Exception as e:
                    logger.debug(f"Failed to parse traits JSON: {e}")
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message="VLM Style Analysis (Raw)",
                            details={"raw_response": response_clean}
                        )
                    )
            else:
                if response_clean.lower().startswith("no"):
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="VLM detected likely hallucination or mismatch in caption.",
                            details={"vlm_reasoning": response_clean},
                            recommendation="Review the caption for accuracy. The VLM suggests a semantic misalignment."
                        )
                    )
                else:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message="VLM verified caption alignment.",
                            details={"vlm_reasoning": response_clean}
                        )
                    )

        except Exception as e:
            logger.warning(f"VLM Judge failed for {sample.path}: {e}")

        return sample

    # ------------------------------------------------------------------
    # Presets mode
    # ------------------------------------------------------------------

    def _process_presets(self, sample: Sample) -> Sample:
        preset_names = self.config.get("presets", [])
        if not preset_names:
            preset_names = list(VLM_PRESETS.keys())

        # Validate preset names
        preset_names = [p for p in preset_names if p in VLM_PRESETS]
        if not preset_names:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        if self._ml_available:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            failed_presets = []
            for preset_name in preset_names:
                labels = VLM_PRESETS[preset_name]
                ok = self._classify_preset(sample, pil_image, preset_name, labels)
                if not ok:
                    failed_presets.append(preset_name)
            # Heuristic fallback for any that failed, reuse already-loaded image
            if failed_presets:
                self._heuristic_presets(sample, image, failed_presets)
        else:
            self._heuristic_presets(sample, image, preset_names)

        return sample

    def _classify_preset(
        self,
        sample: Sample,
        pil_image: Image.Image,
        preset_name: str,
        labels: List[str],
    ) -> bool:
        """Run VLM inference for a single preset. Returns True on success."""
        try:
            import torch

            labels_str = ", ".join(labels)
            prompt = (
                f"USER: <image>\nClassify this image into exactly one of: "
                f"{labels_str}. Answer with just the label.\nASSISTANT:"
            )

            inputs = self._processor(
                prompt, images=pil_image, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=32
                )
                response = self._processor.decode(
                    output[0], skip_special_tokens=True
                )

            response_clean = response.split("ASSISTANT:")[-1].strip()
            matched = self._match_label(response_clean, labels)

            sample.detections.append(
                {
                    "type": "vlm_preset",
                    "preset": preset_name,
                    "label": matched,
                    "raw_response": response_clean,
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Preset {preset_name} VLM failed: {e}")
            return False

    @staticmethod
    def _match_label(response: str, labels: List[str]) -> str:
        response_lower = response.lower().strip()

        # Exact match
        for label in labels:
            if label == response_lower:
                return label

        # Substring match (response contains a label)
        for label in labels:
            if label in response_lower:
                return label

        # Token overlap match
        response_tokens = set(response_lower.replace("_", " ").split())
        best_label = labels[-1]  # default to last (often "neutral" / "none")
        best_overlap = 0
        for label in labels:
            label_tokens = set(label.replace("_", " ").split())
            overlap = len(response_tokens & label_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        return best_label

    def _heuristic_presets(
        self,
        sample: Sample,
        frame_bgr: Optional[np.ndarray],
        preset_names: List[str],
    ) -> None:
        # Load frame if not provided
        if frame_bgr is None:
            frame_bgr = self._load_image(sample)
        if frame_bgr is None:
            return

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())

        for preset_name in preset_names:
            if preset_name not in VLM_PRESETS:
                continue

            labels = VLM_PRESETS[preset_name]

            if preset_name == "time_of_day":
                label = self._heuristic_time_of_day(mean_brightness, labels)
            elif preset_name == "shot_scale":
                label = self._heuristic_shot_scale(gray, labels)
            else:
                # Default to neutral / last label for presets without heuristics
                neutral_candidates = {"neutral", "none_visible", "no_face", "calm"}
                label = next(
                    (lb for lb in labels if lb in neutral_candidates), labels[-1]
                )

            sample.detections.append(
                {
                    "type": "vlm_preset",
                    "preset": preset_name,
                    "label": label,
                    "method": "heuristic",
                }
            )

    @staticmethod
    def _heuristic_time_of_day(mean_brightness: float, labels: List[str]) -> str:
        if mean_brightness < 40:
            return "night" if "night" in labels else labels[-1]
        if mean_brightness < 80:
            return "dusk" if "dusk" in labels else labels[-1]
        if mean_brightness < 140:
            return "morning" if "morning" in labels else labels[0]
        if mean_brightness < 200:
            return "midday" if "midday" in labels else labels[0]
        return "afternoon" if "afternoon" in labels else labels[0]

    @staticmethod
    def _heuristic_shot_scale(gray: np.ndarray, labels: List[str]) -> str:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        if edge_density > 0.15:
            return "extreme_close_up" if "extreme_close_up" in labels else labels[0]
        if edge_density > 0.08:
            return "close_up" if "close_up" in labels else labels[0]
        if edge_density > 0.04:
            return "medium" if "medium" in labels else labels[2]
        if edge_density > 0.02:
            return "full" if "full" in labels else labels[3]
        return "wide" if "wide" in labels else labels[-2]

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None
