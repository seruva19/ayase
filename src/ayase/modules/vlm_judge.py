import logging
import numpy as np
import cv2
from typing import Optional, List
from PIL import Image

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class VLMJudgeModule(PipelineModule):
    """
    Advanced semantic reasoning module using Vision-Language Models (VLM).
    Detects hallucinations in captions and verifies complex spatial/logical relationships.
    """
    name = "vlm_judge"
    description = "Advanced semantic verification using VLM (e.g. LLaVA)"
    default_config = {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "max_new_tokens": 256,
        "mode": "verify",  # 'verify' (hallucination check) or 'traits' (style analysis)
        "verification_prompt": "Does the following text accurately describe the image? Answer only with 'Yes' or 'No' and explain why briefly. Text: '{caption}'",
        "traits_prompt": "Analyze this video and identify its distinctive visual style characteristics. Format your response as a JSON object where keys are trait names and values are scores (0-1). Example: {'warm_tones': 0.8, 'cinematic': 0.9}",
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
        if not sample.caption:
            return sample
            
        if not self._ml_available:
            # Fallback to a simpler check or skip
            return sample

        try:
            import torch
            image = self._load_image(sample)
            if image is None:
                return sample

            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Format prompt based on mode
            mode = self.config.get("mode", "verify")
            if mode == "traits":
                prompt = f"USER: <image>\n{self.config.get('traits_prompt')}\nASSISTANT:"
            else:
                prompt = f"USER: <image>\n{self.verification_prompt.format(caption=sample.caption.text if sample.caption else '')}\nASSISTANT:"
            
            inputs = self._processor(prompt, images=pil_image, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                response = self._processor.decode(output[0], skip_special_tokens=True)
            
            # Parse response
            response_clean = response.split("ASSISTANT:")[-1].strip()
            
            if mode == "traits":
                # Attempt to parse JSON traits
                import json
                import re
                try:
                    # Find JSON block
                    json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                    if json_match:
                        traits = json.loads(json_match.group(0))
                        # Store in detections for selection logic to use later
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
                # Original Verification Mode
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
