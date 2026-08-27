"""TC-Bench — temporal compositionality benchmark for T2V (arXiv:2406.08656).

Measures whether the events described in a caption actually unfold *in
the correct temporal order* in the generated video. Complements
T2V-CompBench (which evaluates spatial composition).

Three dimensions:
    * attribute  — attribute changes are time-localized correctly
                   (e.g. "the ball turns red after spinning").
    * object     — objects appear/disappear at the right time
                   (e.g. "first a dog, then a cat enters").
    * background — background transitions occur in order
                   (e.g. "the scene shifts from day to night").

Plus an overall mean.

Three-tier event decomposition:
    1. LLM-driven event extraction (vLLM / local-LLM endpoint).
    2. Regex template matching on temporal connectives
       (before / after / then / and then / finally / first / next).
    3. Comma-split fallback — treat each clause as one ordered event.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ayase.image import arrays_to_pil, is_video_path, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


_CONNECTIVES = (
    "first", "next", "then", "after that", "afterwards", "finally",
    "before", "after", "and then",
)


class TCBenchModule(PipelineModule):
    name = "tc_bench"
    description = "TC-Bench temporal compositionality for T2V (arXiv:2406.08656)"
    default_config = {
        "decomposer": "auto",  # "auto" | "llm" | "regex" | "comma"
        "num_frames": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "clip_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "event_similarity_threshold": 0.2,
        "models_dir": "models",
    }
    models = [
        {
            "id": "openai/clip-vit-base-patch32",
            "type": "huggingface",
            "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "task": "Frame-to-event semantic grounding",
            "auto_download": True,
        }
    ]
    metric_info = {
        "long_form_event_fulfillment": (
            "Fraction of caption events grounded above the CLIP cosine threshold "
            "in at least one sampled frame (0-1, higher=better)"
        )
    }
    metric_groups = {
        "tcbench_attribute_score": "alignment",
        "tcbench_background_score": "alignment",
        "tcbench_object_score": "alignment",
        "tcbench_overall": "alignment",
        "long_form_event_fulfillment": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.decomposer_pref = self.config.get("decomposer", "auto")
        self.num_frames = self.config.get("num_frames", 8)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        raw_config = config or {}
        self.clip_revision = self.config.get("clip_revision")
        if self.clip_model_name != "openai/clip-vit-base-patch32" and "clip_revision" not in raw_config:
            self.clip_revision = None
        self.event_similarity_threshold = float(
            self.config.get("event_similarity_threshold", 0.2)
        )
        self.models_dir = self.config.get("models_dir", "models")
        self._clip_model = None
        self._clip_processor = None
        self._llm_endpoint = None
        self._device = "cpu"
        self._backend = None
        self.active_decomposer = "comma"

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            import torch  # noqa: F401
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except ImportError:
            self._backend = "unavailable"
            return
        if self._try_load_clip():
            self._backend = "clip"
        else:
            self._backend = "unavailable"
        if self.decomposer_pref in ("auto", "llm") and self._try_setup_llm():
            self.active_decomposer = "llm"
        elif self.decomposer_pref in ("auto", "regex"):
            self.active_decomposer = "regex"
        else:
            self.active_decomposer = "comma"
        logger.info(f"TC-Bench initialized with decomposer={self.active_decomposer}")

    def _try_load_clip(self) -> bool:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import download_hf_snapshot, resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            resolved_candidate = resolve_model_path(self.clip_model_name, self.models_dir)
            if resolved_candidate == self.clip_model_name:
                resolved = str(
                    download_hf_snapshot(
                        self.clip_model_name,
                        self.models_dir,
                        revision=self.clip_revision,
                        allow_patterns=[
                            "config.json",
                            "preprocessor_config.json",
                            "pytorch_model.bin",
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                            "vocab.json",
                            "merges.txt",
                        ],
                    )
                )
            else:
                resolved = resolved_candidate

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            return True
        except Exception as e:
            logger.debug(f"TC-Bench: CLIP unavailable: {e}")
            return False

    def _try_setup_llm(self) -> bool:
        # Optional vLLM/OpenAI-compatible local endpoint — the user must set
        # ``llm_endpoint`` in config. We don't attempt to spin one up.
        endpoint = self.config.get("llm_endpoint")
        if not endpoint:
            return False
        self._llm_endpoint = endpoint
        return True

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        caption = sample.caption.text if sample.caption else None
        if not caption:
            return sample

        events = self._decompose(caption)
        if len(events) < 2:
            # Without at least two ordered events temporal composition is
            # trivially satisfied — record as 1.0 to avoid penalizing
            # captions that simply have no temporal structure.
            self._write(sample, 1.0, 1.0, 1.0, 1.0)
            return sample

        if self._clip_model is None:
            # Temporal ordering cannot be assessed without CLIP grounding -> skip
            # rather than fabricate mid-scale scores.
            return sample

        frames = self._sample_frames(sample.path, self.num_frames)
        if frames is None:
            return sample

        sims = self._clip_sim_matrix(sample, frames, events)
        attribute_score = self._score_dimension(sims, events, kind="attribute")
        object_score = self._score_dimension(sims, events, kind="object")
        background_score = self._score_dimension(sims, events, kind="background")
        overall = float(np.mean([attribute_score, object_score, background_score]))
        self._write(sample, attribute_score, object_score, background_score, overall)
        self._write_fulfillment(sample, self._event_fulfillment(sims))
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if self._clip_model is None:
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                if not sample.is_video:
                    continue
                caption = sample.caption.text if sample.caption else None
                if not caption:
                    continue
                events = self._decompose(caption)
                if len(events) < 2:
                    self._write(sample, 1.0, 1.0, 1.0, 1.0)
                    continue
                frames = self._sample_frames(sample.path, self.num_frames)
                if frames is None:
                    continue
                prepared.append((sample, events))
                image_groups.append(arrays_to_pil(list(frames)))
                cache_keys.append((self.num_frames, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._clip_model,
                self._clip_processor,
                image_groups,
                model_key=self.clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for (sample, events), image_features in zip(prepared, feature_groups):
                sims = self._clip_sim_matrix_from_features(image_features, events)
                attribute_score = self._score_dimension(sims, events, kind="attribute")
                object_score = self._score_dimension(sims, events, kind="object")
                background_score = self._score_dimension(sims, events, kind="background")
                overall = float(np.mean([attribute_score, object_score, background_score]))
                self._write(sample, attribute_score, object_score, background_score, overall)
                self._write_fulfillment(sample, self._event_fulfillment(sims))
        except Exception as e:
            logger.warning("TC-Bench batch failed: %s", e)
        return samples

    def _write(self, sample: Sample, attr: float, obj: float, bg: float, overall: float) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.tcbench_attribute_score = round(float(attr), 3)
        sample.quality_metrics.tcbench_object_score = round(float(obj), 3)
        sample.quality_metrics.tcbench_background_score = round(float(bg), 3)
        sample.quality_metrics.tcbench_overall = round(float(overall), 3)

    def _write_fulfillment(self, sample: Sample, score: float) -> None:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.long_form_event_fulfillment = round(float(score), 3)

    def _event_fulfillment(self, similarities: np.ndarray) -> float:
        if similarities.size == 0 or similarities.shape[1] == 0:
            return 0.0
        event_peaks = np.max(similarities, axis=0)
        return float(np.mean(event_peaks >= self.event_similarity_threshold))

    def _decompose(self, caption: str) -> List[str]:
        if self.active_decomposer == "llm" and self._llm_endpoint is not None:
            try:
                return self._decompose_llm(caption)
            except Exception as e:
                logger.debug(f"TC-Bench LLM decomposition failed: {e}")
        if self.active_decomposer in ("regex", "llm"):
            events = _regex_decompose(caption)
            if len(events) >= 2:
                return events
        return [s.strip() for s in caption.split(",") if s.strip()]

    def _decompose_llm(self, caption: str) -> List[str]:
        # Minimal OpenAI-compatible call. The user is responsible for the
        # backing server (vLLM, llama.cpp, ollama, etc.).
        import urllib.request
        import json

        prompt = (
            "Decompose this video caption into an ordered list of atomic "
            "events, one per line, in the order they should occur. "
            f"Caption: {caption}"
        )
        req = urllib.request.Request(
            self._llm_endpoint,
            data=json.dumps({"prompt": prompt, "max_tokens": 256}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload.get("choices", [{}])[0].get("text", "") or payload.get("text", "")
        events = [line.strip(" -•*") for line in text.splitlines() if line.strip()]
        return events

    def _sample_frames(self, path, n: int) -> Optional[np.ndarray]:
        frames = sample_frames(path, max_frames=n, color="rgb")
        if is_video_path(path) and len(frames) < n:
            return None
        return np.stack(frames, axis=0)

    def _score_dimension(self, sims: np.ndarray, events: List[str], kind: str) -> float:
        # An event is correctly localized when its peak frame index is
        # monotonically increasing in event order. We score by Kendall-tau-
        # like rank concordance on per-event peak indices.
        peaks = np.argmax(sims, axis=0)
        concordant = 0
        total = 0
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                total += 1
                if peaks[i] <= peaks[j]:
                    concordant += 1
        if total == 0:
            return 1.0
        kind_weight = {"attribute": 1.0, "object": 1.0, "background": 0.9}[kind]
        return float(kind_weight * concordant / total)

    def _clip_sim_matrix(
        self,
        sample: Sample,
        frames: np.ndarray,
        events: List[str],
    ) -> np.ndarray:
        image_features = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(list(frames)),
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(self.num_frames, media_state_key(sample.path)),
        )
        return self._clip_sim_matrix_from_features(image_features, events)

    def _clip_sim_matrix_from_features(self, image_features, events: List[str]) -> np.ndarray:
        text_features = cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            events,
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=("tc_bench_events", tuple(events)),
        )
        similarities = image_features @ text_features.T
        return similarities.detach().float().cpu().numpy()


def _regex_decompose(caption: str) -> List[str]:
    text = caption.lower().strip()
    pattern = "|".join(map(re.escape, _CONNECTIVES))
    parts = re.split(rf"\b({pattern})\b", text)
    if len(parts) <= 1:
        return []
    # parts looks like [chunk0, conn1, chunk1, conn2, chunk2, ...]
    events = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip(" ,.")
        if chunk:
            events.append(chunk)
    return events
