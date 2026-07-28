"""VQA² image/video no-reference quality scoring (ACM MM 2025).

VQA² predicts a human-aligned visual quality score from the next-token logits
of its released LLaVA-Qwen scorer. Ayase uses the authors' exact five quality
tokens, ``logits[:, -3]`` position, slow/fast video streams, and any-resolution
image preprocessing. The resulting score is in ``[0.2, 1.0]``; higher is
better.

The Apache-2.0 runtime source, 7B scorer checkpoint, and SlowFast checkpoint
are pinned and downloaded automatically into ``models_dir``. No substitute or
heuristic backend is used.
"""

from __future__ import annotations

import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

VQA2_SOURCE_REVISION = "9087c7952052088a6eb01bac4408bff903ab9e41"
VQA2_MODEL_REVISION = "297de10254d0b4d435db436e1fcaacce5d976fd6"
VQA2_SLOWFAST_REVISION = "8ab5deb746da9139288cbcbf3d155f1c94ff2a8e"
VQA2_SOURCE_URL = (
    "https://github.com/Q-Future/"
    "Visual-Question-Answering-for-Video-Quality-Assessment/archive/"
    f"{VQA2_SOURCE_REVISION}.zip"
)
VQA2_MODEL_ID = "q-future/VQA-UGC-Scorer-llava_qwen"
VQA2_SLOWFAST_ID = "JZHWS/slowfast"

# Published token order: excellent/high, good, fair, poor, bad/low.
VQA2_QUALITY_TOKEN_IDS = (1550, 1661, 6624, 7852, 3347)
VQA2_WA5_WEIGHTS = (1.0, 0.8, 0.6, 0.4, 0.2)
VQA2_LOGIT_POSITION = -3
VQA2_IMAGE_ASSISTANT_PREFIX = "The overall quality of the image is"


def _wa5(logits5: Sequence[float]) -> float:
    """Apply the upstream five-level weighted-average scorer."""
    values = np.asarray(logits5, dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("VQA² requires exactly five finite quality logits")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return float(np.inner(probabilities, np.asarray(VQA2_WA5_WEIGHTS)))


class VQA2Module(PipelineModule):
    """Run the upstream VQA² scorer on images or videos."""

    name = "vqa2"
    description = "VQA² LMM image/video quality score (ACM MM 2025)"
    requires_external_backend = False
    default_config = {
        "model_id": VQA2_MODEL_ID,
        "model_revision": VQA2_MODEL_REVISION,
        "source_revision": VQA2_SOURCE_REVISION,
        "slowfast_revision": VQA2_SLOWFAST_REVISION,
        "models_dir": "models",
        "device": "auto",
        # The upstream protocol processes every frame. Set an integer only as
        # an explicit memory-safety override; uniform sampling preserves span.
        "max_frames": None,
    }
    models = [
        {
            "id": VQA2_MODEL_ID,
            "type": "huggingface",
            "task": "VQA² UGC image/video quality scorer",
            "size": "16.2 GB",
            "vram": "~18 GB",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned revision {VQA2_MODEL_REVISION}",
        },
        {
            "id": VQA2_SLOWFAST_ID,
            "type": "huggingface",
            "task": "SlowFast motion feature extractor used by VQA²",
            "size": "139 MB",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned revision {VQA2_SLOWFAST_REVISION}",
        },
        {
            "id": f"VQA2-source-{VQA2_SOURCE_REVISION}.zip",
            "type": "local",
            "url": VQA2_SOURCE_URL,
            "task": "Pinned upstream VQA² LLaVA runtime source",
            "auto_download": True,
            "notes": "Apache-2.0",
        },
    ]
    metric_info = {
        "vqa2_score": (
            "VQA² five-level no-reference image/video quality score "
            "(0.2–1.0, higher=better)"
        ),
    }
    metric_groups = {"vqa2_score": "nr_quality"}

    def __init__(self, config=None):
        super().__init__(config)
        self.model_id = str(self.config.get("model_id", VQA2_MODEL_ID))
        self.model_revision = str(
            self.config.get("model_revision", VQA2_MODEL_REVISION)
        )
        self.source_revision = str(
            self.config.get("source_revision", VQA2_SOURCE_REVISION)
        )
        self.slowfast_revision = str(
            self.config.get("slowfast_revision", VQA2_SLOWFAST_REVISION)
        )
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.device_config = str(self.config.get("device", "auto"))
        max_frames = self.config.get("max_frames")
        self.max_frames = int(max_frames) if max_frames is not None else None

        self.device = None
        self._backend: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._runtime_root: Optional[Path] = None

    @staticmethod
    def _extract_runtime(archive: Path, destination: Path, revision: str) -> Path:
        """Extract only the upstream inference package and apply a path shim."""
        runtime_root = destination / "quality_scoring"
        marker = destination / ".complete"
        if marker.is_file() and (runtime_root / "llava" / "__init__.py").is_file():
            VQA2Module._patch_runtime(runtime_root)
            return runtime_root

        destination.mkdir(parents=True, exist_ok=True)
        prefix = (
            "Visual-Question-Answering-for-Video-Quality-Assessment-"
            f"{revision}/quality_scoring/"
        )
        root = destination.resolve()
        extracted = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not member.filename.startswith(prefix):
                    continue
                relative = member.filename[len(prefix):]
                if not relative or (
                    not relative.startswith("llava/")
                    and relative not in {"LICENSE", "README.md"}
                ):
                    continue
                if ".." in Path(relative).parts:
                    raise ValueError(
                        f"Unsafe VQA² archive member: {member.filename}"
                    )
                target = (runtime_root / relative).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsafe VQA² archive member: {member.filename}"
                    ) from exc
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                extracted += 1

        if extracted == 0 or not (runtime_root / "llava" / "__init__.py").is_file():
            raise RuntimeError("VQA² runtime was not found in source archive")

        VQA2Module._patch_runtime(runtime_root)
        marker.touch()
        return runtime_root

    @staticmethod
    def _patch_runtime(runtime_root: Path) -> None:
        """Apply two deterministic compatibility fixes to the pinned source."""
        # The upstream SlowFast loader assumes process CWD. Replace only that
        # path lookup; model architecture and scoring code remain unchanged.
        builder = runtime_root / "llava" / "model" / "slowfast" / "builder.py"
        source = builder.read_text(encoding="utf-8")
        old = "torch.load('slowfast.pth',weights_only=False)"
        new = (
            "torch.load(os.environ['AYASE_VQA2_SLOWFAST_PATH'], "
            "weights_only=False)"
        )
        if old in source:
            source = "import os\n" + source.replace(old, new, 1)
            builder.write_text(source, encoding="utf-8")
        elif new not in source:
            raise RuntimeError("Unsupported VQA² SlowFast loader revision")

        # The released image branch refers to an undefined uppercase NUM;
        # the surrounding loop variable is image_idx. The authors' image
        # scorer otherwise fails before producing logits.
        architecture = runtime_root / "llava" / "model" / "llava_arch.py"
        source = architecture.read_text(encoding="utf-8")
        if "image_sizes[NUM]" in source:
            source = source.replace("image_sizes[NUM]", "image_sizes[image_idx]")
            architecture.write_text(source, encoding="utf-8")
        elif "image_sizes[image_idx]" not in source:
            raise RuntimeError("Unsupported VQA² image-size indexing revision")

    def _download_assets(self) -> Tuple[Path, Path, Path]:
        from ayase.config import download_hf_snapshot, download_model_file

        archive = download_model_file(
            f"vqa2/source-{self.source_revision}.zip",
            (
                "https://github.com/Q-Future/"
                "Visual-Question-Answering-for-Video-Quality-Assessment/archive/"
                f"{self.source_revision}.zip"
            ),
            self.models_dir,
        )
        runtime = self._extract_runtime(
            archive,
            Path(self.models_dir).resolve() / "vqa2" / f"source-{self.source_revision}",
            self.source_revision,
        )
        model_path = download_hf_snapshot(
            self.model_id,
            self.models_dir,
            revision=self.model_revision,
            ignore_patterns=["trainer_state.json", "training_args.bin"],
        )
        slowfast_root = download_hf_snapshot(
            VQA2_SLOWFAST_ID,
            self.models_dir,
            revision=self.slowfast_revision,
            allow_patterns=["slowfast.pth"],
        )
        slowfast_path = slowfast_root / "slowfast.pth"
        if not slowfast_path.is_file():
            raise RuntimeError("VQA² SlowFast checkpoint download is incomplete")
        return runtime, model_path, slowfast_path

    @staticmethod
    def _activate_runtime(runtime_root: Path) -> None:
        from ayase._compat import apply_patches

        apply_patches()
        runtime_text = str(runtime_root)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)

        loaded = sys.modules.get("llava")
        if loaded is None:
            return
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        try:
            loaded_file.relative_to(runtime_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Another llava runtime is already loaded from {loaded_file}"
            ) from exc

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch

            if self.device_config not in {"auto", "cuda"}:
                logger.warning("VQA² supports device='auto' or device='cuda'")
                return
            if not torch.cuda.is_available():
                logger.warning("VQA² requires a CUDA GPU for the upstream 7B scorer")
                return

            runtime, model_path, slowfast_path = self._download_assets()
            self._activate_runtime(runtime)
            os.environ["AYASE_VQA2_SLOWFAST_PATH"] = str(slowfast_path)

            from llava.mm_utils import get_model_name_from_path  # type: ignore
            from llava.model.builder import load_pretrained_model  # type: ignore
            from llava.utils import disable_torch_init  # type: ignore

            disable_torch_init()
            model_name = get_model_name_from_path(str(model_path))
            tokenizer, model, image_processor, _context = load_pretrained_model(
                str(model_path),
                None,
                model_name,
                attn_implementation=None,
            )
            if image_processor is None:
                raise RuntimeError(
                    "VQA² upstream loader did not initialise its vision processor"
                )
            model.half()
            model.eval()

            self.device = torch.device("cuda")
            self._tokenizer = tokenizer
            self._model = model
            self._image_processor = image_processor
            self._runtime_root = runtime
            self._backend = "vqa2"
            logger.info("VQA² initialised from %s", model_path)
        except ImportError as exc:
            logger.warning("VQA² dependency is unavailable: %s", exc)
        except Exception as exc:
            logger.warning("VQA² setup failed: %s", exc)
            self._backend = None

    def _build_input_ids(self, user_value: str, assistant_value: str):
        import re

        import torch
        from llava.constants import (  # type: ignore
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        tokenizer = self._tokenizer
        im_start, im_end = tokenizer.additional_special_tokens_ids
        newline = tokenizer("\n").input_ids
        system_role = tokenizer("system").input_ids + newline
        system_message = "You are a helpful assistant."

        input_ids: List[int] = []
        input_ids += (
            [im_start]
            + system_role
            + tokenizer(system_message).input_ids
            + [im_end]
            + newline
        )

        source = [user_value, {"from": "gpt", "value": assistant_value}]
        for index, sentence in enumerate(source):
            role = "<|im_start|>user" if index == 0 else "<|im_start|>assistant"
            if index == 0 and DEFAULT_IMAGE_TOKEN in user_value:
                texts = user_value.split(DEFAULT_IMAGE_TOKEN)
                encoded = tokenizer(role).input_ids + newline
                for text_index, text in enumerate(texts):
                    encoded += tokenizer(text).input_ids
                    if text_index < len(texts) - 1:
                        encoded += [IMAGE_TOKEN_INDEX]
                encoded += [im_end] + newline
                if sum(token == IMAGE_TOKEN_INDEX for token in encoded) != len(
                    re.findall(DEFAULT_IMAGE_TOKEN, user_value)
                ):
                    raise RuntimeError("VQA² image-token prompt construction failed")
            else:
                value = sentence["value"]
                encoded = tokenizer(role).input_ids + newline
                if value is not None:
                    encoded += tokenizer(value).input_ids + [im_end] + newline
            input_ids += encoded

        return torch.tensor([input_ids], dtype=torch.long)

    @staticmethod
    def _logits_tensor(output):
        if hasattr(output, "logits"):
            return output.logits
        return output["logits"]

    def _score_from_logits(self, logits_row) -> float:
        return _wa5([logits_row[token].item() for token in VQA2_QUALITY_TOKEN_IDS])

    def _score_image(self, image) -> float:
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore
        from llava.mm_utils import process_anyres_image  # type: ignore

        patches = process_anyres_image(
            image,
            self._image_processor,
            self._model.config.image_grid_pinpoints,
        )
        patches = patches.half().to(self.device)
        image_tensors = [[patches.repeat(4, 1, 1, 1)], [patches]]
        input_ids = self._build_input_ids(
            DEFAULT_IMAGE_TOKEN, VQA2_IMAGE_ASSISTANT_PREFIX
        ).to(self.device)

        with torch.inference_mode():
            output = self._model(
                input_ids,
                images=image_tensors,
                image_sizes=[image.size],
                modalities=["image"],
            )
        logits = self._logits_tensor(output)[:, VQA2_LOGIT_POSITION]
        return self._score_from_logits(logits.mean(0).float())

    def _load_video(self, path: Path) -> Tuple[List[object], List[object]]:
        from decord import VideoReader, cpu
        from PIL import Image

        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frame_count = len(reader)
        if frame_count == 0:
            return [], []

        indices = np.arange(frame_count, dtype=np.int64)
        if self.max_frames is not None and frame_count > self.max_frames:
            indices = np.linspace(
                0, frame_count - 1, self.max_frames, dtype=np.int64
            )
        arrays = reader.get_batch(indices.tolist()).asnumpy()
        frames = [Image.fromarray(array) for array in arrays]

        fps = max(1, round(float(reader.get_avg_fps())))
        original_fast = set(range(0, frame_count, fps))
        fast_frames = [
            frame for frame, index in zip(frames, indices) if int(index) in original_fast
        ]
        if not fast_frames:
            fast_frames = [frames[0]]
        return frames, fast_frames

    def _score_video(
        self, slow_frames: Sequence[object], fast_frames: Sequence[object]
    ) -> float:
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore

        if not slow_frames:
            raise ValueError("VQA² received an empty video")
        usable = len(slow_frames) // 4 * 4
        if usable:
            slow_frames = slow_frames[:usable]
        else:
            slow_frames = list(slow_frames) + [slow_frames[-1]] * (
                4 - len(slow_frames)
            )

        slow_tensor = self._image_processor.preprocess(
            list(slow_frames), return_tensors="pt"
        )["pixel_values"]
        fast_tensor = self._image_processor.preprocess(
            list(fast_frames), return_tensors="pt"
        )["pixel_values"]
        image_tensors = [
            [slow_tensor.half().to(self.device)],
            [fast_tensor.half().to(self.device)],
        ]
        input_ids = self._build_input_ids(
            DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN, ""
        ).to(self.device)

        with torch.inference_mode():
            output = self._model(input_ids, images=image_tensors)
        logits = self._logits_tensor(output)[:, VQA2_LOGIT_POSITION]
        return self._score_from_logits(logits.mean(0).float())

    @staticmethod
    def _load_image(path: Path):
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB")

    def process(self, sample: Sample) -> Sample:
        if self._backend != "vqa2" or self._model is None:
            return sample

        try:
            if sample.is_video:
                slow_frames, fast_frames = self._load_video(sample.path)
                score = self._score_video(slow_frames, fast_frames)
            else:
                score = self._score_image(self._load_image(sample.path))
            if not np.isfinite(score) or not 0.2 <= score <= 1.0:
                logger.warning("VQA² produced invalid score: %r", score)
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vqa2_score = float(score)
        except Exception as exc:
            logger.warning("VQA² failed for %s: %s", sample.path, exc)
        return sample

    def on_dispose(self) -> None:
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._backend = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        super().on_dispose()
