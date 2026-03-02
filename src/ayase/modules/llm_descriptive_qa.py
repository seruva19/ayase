"""LMM (Large Multimodal Model) Descriptive Quality Assessment module.

Uses large multimodal models (e.g., LLaVA-NeXT, GPT-4V) to generate
natural language quality explanations. Provides interpretable quality
assessment with specific issue identification and reasoning.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LLMDescriptiveQAModule(PipelineModule):
    name = "llm_descriptive_qa"
    description = "LMM-based interpretable quality assessment with explanations"
    default_config = {
        "model_name": "llava-hf/llava-v1.6-mistral-7b-hf",  # LLaVA-NeXT
        "use_openai": False,  # Use OpenAI GPT-4V instead
        "openai_api_key": None,  # OpenAI API key if using GPT-4V
        "num_frames": 4,  # Frames to analyze
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "llava-hf/llava-v1.6-mistral-7b-hf")
        self.use_openai = self.config.get("use_openai", False)
        self.openai_api_key = self.config.get("openai_api_key", None)
        self.num_frames = self.config.get("num_frames", 4)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._model = None
        self._processor = None

    def setup(self) -> None:
        try:
            if self.use_openai:
                # OpenAI API setup
                if self.openai_api_key:
                    self._ml_available = True
                    logger.info("LLM Descriptive QA initialized with OpenAI API")
                else:
                    logger.warning("OpenAI API key not provided")
            else:
                # Local LLaVA model setup
                from transformers import AutoProcessor, LlavaForConditionalGeneration
                import torch

                # Set device
                if self.device_config == "auto":
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.device = torch.device(self.device_config)

                logger.info(f"Loading LLaVA model: {self.model_name}")

                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(self.device)

                self._ml_available = True
                logger.info(f"LLM Descriptive QA initialized with LLaVA on {self.device}")

        except ImportError:
            logger.warning(
                "transformers or torch not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.warning(f"Failed to setup LLM Descriptive QA: {e}")

    def _load_key_frames(self, video_path: Path) -> Optional[List[np.ndarray]]:
        """Load key frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                return None

            # Sample frames uniformly
            num_frames = min(self.num_frames, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()
            return frames if frames else None

        except Exception as e:
            logger.debug(f"Failed to load frames: {e}")
            return None

    def _analyze_with_llava(self, frames: List[np.ndarray]) -> Optional[dict]:
        """Analyze frames using local LLaVA model."""
        try:
            import torch
            from PIL import Image

            # Use middle frame for analysis
            frame = frames[len(frames) // 2]
            image = Image.fromarray(frame)

            # Quality assessment prompt
            prompt = (
                "USER: <image>\nAnalyze the quality of this video frame. "
                "Identify any issues with: sharpness, motion blur, lighting, "
                "artifacts, color, consistency. "
                "Rate the overall quality from 0-100 and explain your assessment.\nASSISTANT:"
            )

            # Process input
            inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            # Generate response
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )

            # Decode response
            response = self._processor.decode(output[0], skip_special_tokens=True)

            # Extract quality score and explanation
            # Simple parsing (production would need better extraction)
            explanation = response.split("ASSISTANT:")[-1].strip()

            # Try to extract numerical score
            import re
            score_match = re.search(r'(\d+)\s*(?:/\s*100)?', explanation)
            if score_match:
                quality_score = float(score_match.group(1))
            else:
                quality_score = 50.0  # Default

            return {
                "quality_score": quality_score,
                "explanation": explanation,
                "issues": self._extract_issues(explanation),
            }

        except Exception as e:
            logger.warning(f"LLaVA analysis failed: {e}")
            return None

    def _analyze_with_openai(self, frames: List[np.ndarray]) -> Optional[dict]:
        """Analyze frames using OpenAI GPT-4V API."""
        try:
            from openai import OpenAI
            import base64
            from io import BytesIO
            from PIL import Image

            client = OpenAI(api_key=self.openai_api_key)

            # Use middle frame
            frame = frames[len(frames) // 2]
            image = Image.fromarray(frame)

            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # API call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze the quality of this video frame. "
                                "Identify any issues with: sharpness, motion, lighting, "
                                "artifacts, color. Rate quality 0-100 and explain.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            explanation = response.choices[0].message.content

            # Extract quality score
            import re
            score_match = re.search(r'(\d+)\s*(?:/\s*100)?', explanation)
            quality_score = float(score_match.group(1)) if score_match else 50.0

            return {
                "quality_score": quality_score,
                "explanation": explanation,
                "issues": self._extract_issues(explanation),
            }

        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            return None

    def _extract_issues(self, explanation: str) -> List[str]:
        """Extract quality issues from explanation text."""
        issues = []
        issue_keywords = ["blur", "artifact", "noise", "dark", "bright", "saturated", "compressed"]

        explanation_lower = explanation.lower()
        for keyword in issue_keywords:
            if keyword in explanation_lower:
                issues.append(keyword)

        return issues

    def process(self, sample: Sample) -> Sample:
        """Process sample with LMM descriptive quality assessment."""
        if not self._ml_available:
            return sample

        try:
            # Load frames
            if sample.is_video:
                frames = self._load_key_frames(sample.path)
            else:
                # Load single image
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
                else:
                    frames = None

            if frames is None:
                return sample

            # Analyze with LMM
            if self.use_openai:
                result = self._analyze_with_openai(frames)
            else:
                result = self._analyze_with_llava(frames)

            if result is None:
                return sample

            # Store confidence score (LMM's quality rating)
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.confidence_score = result["quality_score"] / 100.0

            # Add explanation as validation issue
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="LMM Quality Assessment",
                    details={
                        "explanation": result["explanation"],
                        "detected_issues": result["issues"],
                        "quality_score": result["quality_score"],
                    },
                    recommendation=result["explanation"],
                )
            )

            logger.debug(
                f"LMM analysis for {sample.path.name}: {result['quality_score']:.1f}/100"
            )

        except Exception as e:
            logger.warning(f"LLM Descriptive QA processing failed for {sample.path}: {e}")

        return sample
