"""Memory-VQA -- Video Quality Based on Human Memory System.

Neurocomputing 2025 -- models stages of human memory formation (sensory input,
encoding, storage, retrieval, decision) with trained modules for quality
perception.

Ayase does not ship (and cannot currently locate a loadable checkpoint for) the
real Memory-VQA weights. Per the project's no-heuristic policy a named-metric
field must be produced by the real model or left ``None`` -- it must never be
fabricated from an ImageNet ResNet backbone bolted to randomly-initialised,
untrained sensory/encoding/gate/decision heads (which is all the previous
implementation did). Until a genuine Memory-VQA checkpoint is wired in,
``memoryvqa_score`` is left unset.

memoryvqa_score -- higher = better quality (0-1); real model only

REVIVAL NOTES (requires_external_backend -- no turnkey backend)
Metric: Memory-VQA (Neurocomputing 2025).
Category: TRAINING-ONLY.
Why requires_external_backend: Code on GitHub but README says "weights after acceptance"; no loadable checkpoint.
To revive: Reimplement the human-memory HVS model; train on the 5 public UGC-VQA sets; validate you
  reproduce the paper's SRCC/PLCC before flipping requires_external_backend=False. Incremental gains.
Source: Memory-VQA, Neurocomputing 2025 (weights pending acceptance).
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MemoryVQAModule(PipelineModule):
    name = "memoryvqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "Memory-VQA human memory system VQA (Neurocomputing 2025; real model only, disabled if unavailable)"
    default_config = {
        "subsample": 12,
        "memory_size": 8,
    }
    metric_groups = {
        "memoryvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self.memory_size = self.config.get("memory_size", 8)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable Memory-VQA checkpoint is wired in. The previous
        # implementation ran an ImageNet ResNet through untrained, random-init
        # memory-system heads, so its scores were meaningless. Disable rather
        # than fabricate.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "Memory-VQA: no real Memory-VQA model/weights available; module "
            "disabled (memoryvqa_score left unset). Wire a genuine Memory-VQA "
            "checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave memoryvqa_score None (graceful).
        return sample
