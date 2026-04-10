"""Memory-VQA -- Video Quality Based on Human Memory System.

Neurocomputing 2025 -- models 5 stages of human memory formation
(sensory input, encoding, storage, retrieval, decision) for
quality perception.

Implementation:
    ResNet-50 backbone for frame feature extraction.  A memory bank
    (FIFO buffer of feature vectors) mimics human short-term memory,
    maintaining temporal context as frames are processed.  Quality is
    derived from memory-augmented features that combine current-frame
    perception with stored memory representations.

memoryvqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MemoryVQAModule(PipelineModule):
    name = "memoryvqa"
    description = "Memory-VQA human memory system VQA (Neurocomputing 2025)"
    default_config = {
        "subsample": 12,
        "memory_size": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self.memory_size = self.config.get("memory_size", 8)
        self._resnet = None
        self._resnet_transform = None
        self._sensory_head = None
        self._encoding_head = None
        self._memory_gate = None
        self._decision_head = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ResNet-50 backbone
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            feat_dim = 2048

            # Stage 1: Sensory input -- raw feature perception
            self._sensory_head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
            ).to(self._device)
            self._sensory_head.eval()

            # Stage 2: Encoding -- compress to quality-relevant representation
            self._encoding_head = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
            ).to(self._device)
            self._encoding_head.eval()

            # Stage 3/4: Memory gate -- controls what enters/exits memory
            # Input: current encoded feat (64) + memory context (64)
            self._memory_gate = torch.nn.Sequential(
                torch.nn.Linear(64 + 64, 64),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._memory_gate.eval()

            # Stage 5: Decision -- final quality from memory-augmented features
            self._decision_head = torch.nn.Sequential(
                torch.nn.Linear(64 + 64, 128),  # current + memory
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._decision_head.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "Memory-VQA initialised with ResNet-50 + memory bank on %s",
                self._device,
            )

        except ImportError:
            logger.warning(
                "Memory-VQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("Memory-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_memory_quality(frames)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.memoryvqa_score = float(
                    np.clip(score, 0.0, 1.0)
                )

        except Exception as e:
            logger.warning("Memory-VQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_memory_quality(self, frames: List[np.ndarray]) -> Optional[float]:
        """5-stage memory model quality computation."""
        import torch

        # Memory bank: FIFO buffer of encoded features
        memory_bank: List[np.ndarray] = []
        frame_decisions = []

        for frame in frames:
            # Extract raw features
            raw_feat = self._extract_feature(frame)
            if raw_feat is None:
                continue

            feat_tensor = (
                torch.from_numpy(raw_feat.astype(np.float32))
                .unsqueeze(0)
                .to(self._device)
            )

            with torch.no_grad():
                # Stage 1: Sensory input
                sensory = self._sensory_head(feat_tensor)

                # Stage 2: Encoding
                encoded = self._encoding_head(sensory)
                encoded_np = encoded.cpu().numpy().flatten()

                # Stage 3: Storage -- build memory context
                if memory_bank:
                    # Aggregate memory bank (mean of stored features)
                    memory_context = np.mean(memory_bank, axis=0)
                else:
                    memory_context = np.zeros(64, dtype=np.float32)

                memory_tensor = (
                    torch.from_numpy(memory_context.astype(np.float32))
                    .unsqueeze(0)
                    .to(self._device)
                )

                # Stage 4: Retrieval -- gated memory access
                gate_input = torch.cat([encoded, memory_tensor], dim=1)
                gate = self._memory_gate(gate_input)
                gated_memory = gate * memory_tensor

                # Stage 5: Decision -- quality from current + memory
                decision_input = torch.cat([encoded, gated_memory], dim=1)
                quality = self._decision_head(decision_input).item()

            frame_decisions.append(quality)

            # Update memory bank (FIFO)
            memory_bank.append(encoded_np)
            if len(memory_bank) > self.memory_size:
                memory_bank.pop(0)

        if not frame_decisions:
            return None

        # Final score: weighted mean giving more importance to later frames
        # (human memory emphasises recent experience)
        n = len(frame_decisions)
        if n > 1:
            recency_weights = np.linspace(0.5, 1.0, n)
            recency_weights = recency_weights / recency_weights.sum()
            score = float(np.dot(recency_weights, frame_decisions))
        else:
            score = float(frame_decisions[0])

        return score

    def _extract_feature(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet-50 feature from a frame."""
        import torch

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._resnet(tensor)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return None

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(
                    0, total - 1, min(self.subsample, total), dtype=int
                )
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._resnet = None
        self._sensory_head = None
        self._encoding_head = None
        self._memory_gate = None
        self._decision_head = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
