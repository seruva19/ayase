"""Audio-text semantic alignment verification using CLAP.

Computes cosine similarity between audio embeddings and caption text.
Low alignment scores indicate mismatched audio-description pairs."""

import logging
import numpy as np
from typing import Optional
import os

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class AudioTextAlignmentModule(PipelineModule):
    """
    Verifies semantic alignment between audio track and text caption using CLAP.
    Ensures that if the caption mentions sound, the audio actually contains it.
    """
    name = "audio_text_alignment"
    description = "Multimodal alignment check (Audio-Text) using CLAP"
    default_config = {
        "alignment_threshold": 0.2,
        "model_name": "laion/clap-htsat-fused",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.alignment_threshold = self.config.get("alignment_threshold", 0.2)
        self.model_name = self.config.get("model_name", "laion/clap-htsat-fused")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLAP for Audio-Text alignment on {self._device}...")
            
            models_dir = self.config.get("models_dir", "models")
            
            self._model = ClapModel.from_pretrained(self.model_name, cache_dir=models_dir).to(self._device)
            self._processor = ClapProcessor.from_pretrained(self.model_name, cache_dir=models_dir)
            
            self._ml_available = True
            
        except ImportError:
            logger.warning("Transformers not installed. CLAP alignment disabled.")
        except Exception as e:
            logger.warning(f"Failed to setup CLAP: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.caption:
            return sample

        # Check if audio is even present
        # This module assumes an audio track exists if we are checking alignment
        # We can use the audio metadata or just try to load it
        
        try:
            import torch
            import librosa

            # Load audio from video file
            audio, sr = librosa.load(str(sample.path), sr=48000, mono=True, duration=10)
            
            if len(audio) == 0:
                return sample # No audio to align

            # Process with CLAP
            inputs = self._processor(
                text=[sample.caption.text], 
                audios=[audio], 
                return_tensors="pt", 
                padding=True,
                sampling_rate=48000
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                audio_embeds = outputs.audio_embeds
                text_embeds = outputs.text_embeds

                # Normalize
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                similarity = (audio_embeds * text_embeds).sum(dim=-1).item()

            if similarity < self.alignment_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Audio-Text Alignment (Similarity: {similarity:.2f})",
                        details={"clap_similarity": float(similarity)},
                        recommendation="The audio track does not seem to match the textual description. This can occur with generic background music or desynchronized voiceovers."
                    )
                )

        except Exception as e:
            logger.warning(f"CLAP alignment failed for {sample.path}: {e}")

        return sample
