import logging
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class CommonsenseModule(PipelineModule):
    name = "commonsense"
    description = "Checks for common sense violations using VQA or Text Logic"
    default_config = {
        "model_name": "dandelin/vilt-b32-finetuned-vqa",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "dandelin/vilt-b32-finetuned-vqa")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def on_mount(self) -> None:
        super().on_mount()
        try:
            import torch
            from transformers import ViltProcessor, ViltForQuestionAnswering

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading VQA ({self.model_name}) on {self._device}...")

            models_dir = self.config.get("models_dir", "models")

            self._processor = ViltProcessor.from_pretrained(self.model_name, cache_dir=models_dir)
            self._model = ViltForQuestionAnswering.from_pretrained(
                self.model_name, cache_dir=models_dir, use_safetensors=True
            ).to(self._device)
            
            self._ml_available = True
            
        except Exception as e:
            logger.warning(f"Failed to setup Commonsense VQA: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
            
        try:
            import cv2
            from PIL import Image
            
            image = self._load_image(sample)
            if image is None:
                return sample
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Ask a generic question about realism/correctness
            # ViLT is trained on VQA v2, so simple questions work best.
            # "Is this photo real?" might be too abstract.
            # "Is there a person?" is too specific.
            # Let's check for specific artifacts or oddities if we can.
            
            questions = [
                ("Is this a real photo?", "no", "Model thinks this is NOT a real photo"),
                ("Is this a cartoon?", "yes", "Model classifies this as cartoon/animated content"),
                ("Is there a person in the image?", None, None),  # informational
                ("Is this image blurry?", "yes", "Model thinks this image is blurry"),
                ("Is there text in the image?", "yes", "Model detected text/overlay in the image"),
            ]

            answers = {}
            for q, flag_answer, flag_msg in questions:
                encoding = self._processor(pil_image, q, return_tensors="pt").to(self._device)
                outputs = self._model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer = self._model.config.id2label[idx]
                answers[q] = answer

                if flag_answer is not None and answer.lower() == flag_answer:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Commonsense: {flag_msg} (answer='{answer}')",
                            details={"question": q, "answer": answer},
                        )
                    )

            # Cross-question reasoning: real photo + blurry = low quality capture
            if answers.get("Is this a real photo?", "").lower() == "yes" and answers.get("Is this image blurry?", "").lower() == "yes":
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Commonsense: Real photo that is blurry — likely poor quality capture",
                        details={"answers": answers},
                    )
                )
                
        except Exception as e:
            logger.warning(f"Commonsense check failed: {e}")
        
        return sample

    def _load_image(self, sample: Sample):
        try:
            import cv2
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
