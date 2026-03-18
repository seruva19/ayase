"""Common sense adherence module — VBench-2.0 dimension.

Checks if generated images/videos follow common sense rules (object
placement, interaction plausibility, layout consistency).

Backend tiers:
  1. **VLM** — LLaVA-1.5-7b with structured scoring prompt
  2. **ViLT** — ViLT VQA with 5 diagnostic questions → numeric score
  3. **Heuristic** — Color distribution + spatial frequency analysis
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CommonsenseModule(PipelineModule):
    name = "commonsense"
    description = "Common sense adherence (VLM / ViLT VQA / heuristic)"
    default_config = {
        "model_name": "dandelin/vilt-b32-finetuned-vqa",
        "vlm_model": "llava-hf/llava-1.5-7b-hf",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._vlm_model = None
        self._vlm_processor = None
        self._vilt_model = None
        self._vilt_processor = None
        self._device = "cpu"

    def setup(self) -> None:
        # Tier 1: LLaVA VLM
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            vlm_name = self.config.get("vlm_model", "llava-hf/llava-1.5-7b-hf")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")
            dtype = torch.float16 if self._device == "cuda" else torch.float32

            self._vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
                vlm_name, torch_dtype=dtype, cache_dir=models_dir, low_cpu_mem_usage=True,
            ).to(self._device)
            self._vlm_model.eval()
            self._vlm_processor = LlavaNextProcessor.from_pretrained(vlm_name, cache_dir=models_dir)
            self._backend = "vlm"
            logger.info("Commonsense loaded LLaVA on %s", self._device)
            return
        except Exception as e:
            logger.info("VLM unavailable for commonsense: %s", e)

        # Tier 2: ViLT VQA
        try:
            import torch
            from transformers import ViltProcessor, ViltForQuestionAnswering

            vilt_name = self.config.get("model_name", "dandelin/vilt-b32-finetuned-vqa")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")

            self._vilt_processor = ViltProcessor.from_pretrained(vilt_name, cache_dir=models_dir)
            self._vilt_model = ViltForQuestionAnswering.from_pretrained(
                vilt_name, cache_dir=models_dir, use_safetensors=True,
            ).to(self._device)
            self._backend = "vilt"
            logger.info("Commonsense loaded ViLT VQA on %s", self._device)
            return
        except Exception as e:
            logger.info("ViLT unavailable for commonsense: %s", e)

        logger.info("Commonsense using heuristic backend")

    def process(self, sample: Sample) -> Sample:
        image = self._load_image(sample)
        if image is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            if self._backend == "vlm":
                score, issues = self._compute_vlm(image)
            elif self._backend == "vilt":
                score, issues = self._compute_vilt(image)
            else:
                score, issues = self._compute_heuristic(image)

            if score is not None:
                sample.quality_metrics.commonsense_score = score

            for issue in issues:
                sample.validation_issues.append(issue)

        except Exception as e:
            logger.warning("Commonsense check failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Tier 1: VLM (LLaVA)                                                 #
    # ------------------------------------------------------------------ #

    def _compute_vlm(self, image: np.ndarray) -> tuple:
        import torch
        import json
        import re
        from PIL import Image

        issues = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = (
            "USER: <image>\nRate the following aspects of this image on a scale of 1-5:\n"
            "1. Location plausibility (do objects appear in plausible locations?)\n"
            "2. Interaction sense (do interactions between objects/people make sense?)\n"
            "3. Layout consistency (is the spatial layout logically consistent?)\n"
            "Respond ONLY with a JSON object: {\"location\": N, \"interaction\": N, \"layout\": N}\n"
            "ASSISTANT:"
        )

        inputs = self._vlm_processor(prompt, images=pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._vlm_model.generate(**inputs, max_new_tokens=64)
            response = self._vlm_processor.decode(output[0], skip_special_tokens=True)

        response_clean = response.split("ASSISTANT:")[-1].strip()

        try:
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
                loc = float(scores.get("location", 3))
                inter = float(scores.get("interaction", 3))
                layout = float(scores.get("layout", 3))
                # Normalize 1-5 → 0-1
                score = (loc + inter + layout) / 15.0
                return float(np.clip(score, 0.0, 1.0)), issues
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse yes/no or numeric from response
        response_lower = response_clean.lower()
        if "no" in response_lower or "poor" in response_lower or "bad" in response_lower:
            return 0.3, issues
        return 0.7, issues

    # ------------------------------------------------------------------ #
    # Tier 2: ViLT VQA                                                     #
    # ------------------------------------------------------------------ #

    def _compute_vilt(self, image: np.ndarray) -> tuple:
        from PIL import Image

        issues = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        questions = [
            ("Is this a real photo?", "no", "Model thinks this is NOT a real photo"),
            ("Is this a cartoon?", "yes", "Model classifies this as cartoon/animated content"),
            ("Is there a person in the image?", None, None),
            ("Is this image blurry?", "yes", "Model thinks this image is blurry"),
            ("Is there text in the image?", "yes", "Model detected text/overlay in the image"),
        ]

        correct_count = 0
        answers = {}
        for q, flag_answer, flag_msg in questions:
            encoding = self._vilt_processor(pil_image, q, return_tensors="pt").to(self._device)
            outputs = self._vilt_model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self._vilt_model.config.id2label[idx]
            answers[q] = answer

            # Score: "correct" answers contribute positively
            if flag_answer is not None:
                if answer.lower() != flag_answer:
                    correct_count += 1
                else:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Commonsense: {flag_msg} (answer='{answer}')",
                            details={"question": q, "answer": answer},
                        )
                    )
            else:
                correct_count += 1  # Informational questions always count

        # Cross-question reasoning
        if (answers.get("Is this a real photo?", "").lower() == "yes" and
                answers.get("Is this image blurry?", "").lower() == "yes"):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Commonsense: Real photo that is blurry — likely poor quality capture",
                    details={"answers": answers},
                )
            )

        score = correct_count / len(questions)
        return float(score), issues

    # ------------------------------------------------------------------ #
    # Tier 3: Heuristic                                                    #
    # ------------------------------------------------------------------ #

    def _compute_heuristic(self, image: np.ndarray) -> tuple:
        issues = []
        h, w = image.shape[:2]

        # Color distribution analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Natural images have diverse hue distribution
        hue_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        hue_hist = hue_hist.flatten() / (h * w)
        hue_entropy = float(-np.sum(hue_hist[hue_hist > 0] * np.log2(hue_hist[hue_hist > 0] + 1e-10)))
        # Normalize to 0-1 (max entropy for 18 bins = log2(18) ≈ 4.17)
        hue_score = min(hue_entropy / 4.17, 1.0)

        # Spatial frequency — natural images follow 1/f power law
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(dft))
        # Radial average of power spectrum
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = min(cy, cx)
        radial_mean = np.zeros(max_r)
        for ri in range(1, max_r):
            mask = r == ri
            if mask.any():
                radial_mean[ri] = magnitude[mask].mean()

        # Check power law: log(power) vs log(freq) should be approximately linear
        valid = radial_mean[1:] > 0
        if valid.sum() > 10:
            freqs = np.arange(1, max_r)
            log_freq = np.log(freqs[valid])
            log_power = np.log(radial_mean[1:][valid])
            # Linear fit
            coeffs = np.polyfit(log_freq, log_power, 1)
            slope = coeffs[0]
            # Natural images have slope around -2
            deviation = abs(slope - (-2.0))
            freq_score = max(0.0, 1.0 - deviation / 3.0)
        else:
            freq_score = 0.5

        score = 0.5 * hue_score + 0.5 * freq_score
        return float(np.clip(score, 0.0, 1.0)), issues

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

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
