"""Approximate multi-source image-edit evaluation with a hosted vision judge.

This adapter uses a generic rubric inspired by the three MIEScore evaluation
dimensions. It does not use the paper's trained LoRA regression head, weights,
or author prompt, so its 0-100 scores are uncalibrated and are not comparable
to official MIEScore results. Higher is better for every output.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._reward_utils import image_to_data_url, load_rgb_image

logger = logging.getLogger(__name__)

PAPER_URL = "https://arxiv.org/abs/2608.02059"


class MIEScoreModule(PipelineModule):
    """Approximate MIEScore-style evaluation for multi-source image edits."""

    name = "miescore"
    implementation_status = "approximate"
    implementation_notes = "Own hosted-VLM rubric replaces trained LoRA regression head; explicit approximate output fields accurately signal lack of published-method reproduction."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    description = "Approximate three-dimensional multi-source image-edit evaluation"
    input_type = "img"
    provenance = "adapted"
    sources = {"*": f"MIEScore paper — {PAPER_URL}"}
    deviations = {
        "*": "Uses an original generic prompt rubric and hosted vision model instead of the "
        "paper's trained LoRA regression head; outputs are uncalibrated and do not claim "
        "official implementation parity."
    }
    default_config = {
        "model": "gpt-4.1",
        "api_key": None,
        "base_url": None,
        "source_paths": None,
        "instruction": None,
        "timeout": 120.0,
        "max_tokens": 300,
    }
    models = [
        {
            "id": "gpt-4.1",
            "type": "other",
            "url": "https://developers.openai.com/api/docs/models/gpt-4.1",
            "task": "Hosted multimodal rubric judge (configured model: gpt-4.1 by default)",
            "auto_download": False,
            "notes": "Requires API credentials; no paper weights or trained regression head.",
        }
    ]
    metric_info = {
        "miescore_approx_visual_quality": "Approximate visual quality (0-100, higher=better)",
        "miescore_approx_instruction_following": (
            "Approximate edit-instruction following (0-100, higher=better)"
        ),
        "miescore_approx_attribute_preservation": (
            "Approximate source-attribute preservation (0-100, higher=better)"
        ),
    }
    metric_groups = {
        "miescore_approx_visual_quality": "nr_quality",
        "miescore_approx_instruction_following": "alignment",
        "miescore_approx_attribute_preservation": "alignment",
    }
    _output_fields = tuple(metric_info)

    def __init__(self, config=None):
        super().__init__(config)
        self._client: Any = None
        self._backend: Optional[str] = None

    def setup(self) -> None:
        """Create the API client without making a network request."""
        if self.test_mode:
            return
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if self.config.get("api_key"):
                kwargs["api_key"] = self.config["api_key"]
            if self.config.get("base_url"):
                kwargs["base_url"] = self.config["base_url"]
            kwargs["timeout"] = float(self.config.get("timeout", 120.0))
            self._client = OpenAI(**kwargs)
            self._backend = "openai_compatible"
        except Exception as exc:
            logger.warning("MIEScore approximate API client unavailable: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self.test_mode:
            return sample

        self._clear_outputs(sample)
        if self._client is None or sample.is_video:
            return sample

        try:
            source_paths = self._source_paths(sample)
            instruction = self._instruction(sample)
            target_path = Path(sample.path)
            if not source_paths or not instruction or not target_path.is_file():
                return sample
            if any(not path.is_file() for path in source_paths):
                return sample

            content: list[dict[str, Any]] = [
                {"type": "text", "text": self._rubric(instruction, len(source_paths))}
            ]
            for index, path in enumerate(source_paths, start=1):
                content.extend(self._image_parts(f"SOURCE IMAGE {index}", path))
            content.extend(self._image_parts("EDITED TARGET IMAGE", target_path))

            response = self._client.chat.completions.create(
                model=str(self.config["model"]),
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=int(self.config.get("max_tokens", 300)),
            )
            scores = self._parse_response(response.choices[0].message.content)
            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.miescore_approx_visual_quality = scores[0]
            sample.quality_metrics.miescore_approx_instruction_following = scores[1]
            sample.quality_metrics.miescore_approx_attribute_preservation = scores[2]
            sample.metadata["miescore"] = {
                "implementation_status": self.implementation_status,
                "backend": self._backend,
                "model": str(self.config["model"]),
                "source_count": len(source_paths),
                "source_paths": [str(path) for path in source_paths],
                "paper": PAPER_URL,
            }
        except Exception as exc:
            logger.warning("MIEScore approximate evaluation failed for %s: %s", sample.path, exc)
        return sample

    def _clear_outputs(self, sample: Sample) -> None:
        """Remove results and lineage from an earlier evaluation attempt."""
        sample.metadata.pop("miescore", None)
        metrics = sample.quality_metrics
        if metrics is None:
            return
        for field in self._output_fields:
            if hasattr(metrics, field):
                setattr(metrics, field, None)
        provenance = getattr(metrics, "metric_provenance", None)
        if isinstance(provenance, dict):
            for field in self._output_fields:
                provenance.pop(field, None)
        backends = getattr(metrics, "metric_backends", None)
        if isinstance(backends, dict):
            backends.pop(self.name, None)

    def teardown(self) -> None:
        client, self._client = self._client, None
        self._backend = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("MIEScore API client cleanup failed: %s", exc)

    def _source_paths(self, sample: Sample) -> list[Path]:
        raw = self.config.get("source_paths")
        if raw is None:
            raw = sample.metadata.get("source_paths")
        if not isinstance(raw, (list, tuple)) or not raw:
            return []
        paths: list[Path] = []
        for value in raw:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                return []
            paths.append(Path(value))
        return paths

    def _instruction(self, sample: Sample) -> str:
        caption = getattr(sample.caption, "text", None) if sample.caption is not None else None
        value = caption if isinstance(caption, str) and caption.strip() else self.config.get("instruction")
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _image_parts(label: str, path: Path) -> list[dict[str, Any]]:
        image = load_rgb_image(path)
        return [
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
        ]

    @staticmethod
    def _rubric(instruction: str, source_count: int) -> str:
        return (
            "Evaluate an edited image made from the ordered source images below. "
            f"There are {source_count} source images. The edit instruction is: {instruction!r}. "
            "Score three independent dimensions from 0 to 100: visual_quality judges technical "
            "and perceptual finish of the edited target; instruction_following judges whether the "
            "requested transformation and composition were achieved; attribute_preservation judges "
            "whether relevant identities, objects, appearance, and relationships from all source "
            "images were retained when the instruction did not ask to change them. Inspect every "
            "source independently and do not replace the ordered set with a single reference. "
            "Return only a JSON object with exactly these numeric keys: visual_quality, "
            "instruction_following, attribute_preservation."
        )

    @staticmethod
    def _parse_response(text: Any) -> Optional[tuple[float, float, float]]:
        if not isinstance(text, str):
            return None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        keys = ("visual_quality", "instruction_following", "attribute_preservation")
        if not isinstance(payload, dict) or set(payload) != set(keys):
            return None
        values = []
        for key in keys:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                return None
            values.append(number)
        return values[0], values[1], values[2]
