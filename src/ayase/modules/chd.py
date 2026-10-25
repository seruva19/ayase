"""Released CHD-1D over TiTok-L-32 codebook unigram histograms.

Compares independent generated and reference image collections using the released
ImageNet TiTok-L-32 tokenizer. The score is Hellinger distance in [0, 1], where
lower is better. This is the released unigram metric, not the paper's unavailable
CHD-1D/CHD-2D composite or its retrained tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ayase.config import download_hf_snapshot
from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

SOURCE_REVISION = "06c3f986cb37188a59596ba6fc2e6ad6cc5665d5"
SOURCE_ARCHIVE = "https://codeload.github.com/zexiJia/1d-Distance/zip/" + SOURCE_REVISION
MODEL_ID = "yucornetto/tokenizer_titok_l32_imagenet"
MODEL_REVISION = "1c9a2084c59112fd415b7ed97d4c200e864a95de"
MODEL_SHA256 = "fecb8ed7aa6f2a620269b0ad141098ba1628f4a41290b6057dd91dbf5e9bf840"
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_REQUIRED_SOURCE = (
    "modeling/__init__.py",
    "modeling/titok.py",
    "modeling/modules/base_model.py",
    "modeling/modules/blocks.py",
    "modeling/modules/maskgit_vqgan.py",
    "modeling/quantizer/quantizer.py",
)


def _hellinger_distance(real: Sequence[float], generated: Sequence[float]) -> float:
    """Independent canonical implementation of the released CHD-1D reduction."""
    left_values = list(real)
    right_values = list(generated)
    if len(left_values) != len(right_values) or len(left_values) == 0:
        raise ValueError("CHD histograms must be non-empty and have equal lengths")
    if any(not math.isfinite(float(value)) for value in left_values + right_values):
        raise ValueError("CHD histograms must contain only finite values")
    left = [max(0.0, float(value)) for value in left_values]
    right = [max(0.0, float(value)) for value in right_values]
    left_total, right_total = sum(left), sum(right)
    if left_total <= 0 or right_total <= 0:
        raise ValueError("CHD histograms must each contain positive mass")
    squared = sum(
        (math.sqrt(a / left_total) - math.sqrt(b / right_total)) ** 2 for a, b in zip(left, right)
    )
    return math.sqrt(squared) / math.sqrt(2.0)


def _verify_source(root: Path, *, managed: bool) -> None:
    if any(not (root / relative).is_file() for relative in _REQUIRED_SOURCE):
        raise ValueError(f"Incomplete CHD source tree: {root}")
    marker = root / ".ayase-source-revision"
    if managed and not marker.is_file():
        raise ValueError(f"Unverified CHD source tree: {root}")
    if marker.is_file() and marker.read_text(encoding="ascii").strip() != SOURCE_REVISION:
        raise ValueError(f"CHD source revision mismatch: {root}")


def _ensure_source(destination: Path) -> Path:
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if destination.is_dir():
            _verify_source(destination, managed=True)
            return destination
        with tempfile.TemporaryDirectory(prefix="chd-source-", dir=destination.parent) as name:
            work = Path(name)
            archive = work / "source.zip"
            urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
            unpacked = work / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                roots = {
                    Path(item.filename).parts[0] for item in members if Path(item.filename).parts
                }
                if len(roots) != 1:
                    raise ValueError("CHD archive must contain one root")
                base = unpacked.resolve()
                for item in members:
                    target = (unpacked / item.filename).resolve()
                    if target != base and base not in target.parents:
                        raise ValueError(f"Unsafe CHD archive member: {item.filename}")
                    if item.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError(f"CHD archive contains a symlink: {item.filename}")
                bundle.extractall(unpacked)
            extracted = unpacked / roots.pop()
            (extracted / ".ayase-source-revision").write_text(SOURCE_REVISION, encoding="ascii")
            _verify_source(extracted, managed=True)
            extracted.replace(destination)
    return destination


_WORKER = r"""
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.loads(Path(args.input).read_text(encoding="utf-8"))
result = {"ok": False}
try:
    source = Path(request["source_root"]).resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from modeling.titok import TiTok

    device_name = request["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads((Path(request["model_root"]) / "config.json").read_text())
    model = TiTok(config)
    state = load_file(str(Path(request["model_root"]) / "model.safetensors"), device="cpu")
    expected, actual = set(model.state_dict()), set(state)
    if expected != actual:
        raise ValueError("TiTok checkpoint keys mismatch: missing=%d unexpected=%d" %
                         (len(expected - actual), len(actual - expected)))
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device_name)

    def histogram(paths):
        counts = torch.zeros(4096, dtype=torch.float64, device=device_name)
        batch_size = request["batch_size"]
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                images = []
                for path in paths[start:start + batch_size]:
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
                    images.append(torch.from_numpy(array).permute(2, 0, 1))
                batch = torch.stack(images).to(device_name)
                indices = model.encode(batch)[1]["min_encoding_indices"]
                if indices.ndim > 2:
                    indices = indices.squeeze(1)
                if indices.ndim != 2 or indices.shape[1] != 32:
                    raise ValueError("TiTok produced an unexpected token shape: %s" % (tuple(indices.shape),))
                if indices.numel() and (indices.min() < 0 or indices.max() >= 4096):
                    raise ValueError("TiTok produced an out-of-range codebook index")
                counts += torch.bincount(indices.reshape(-1), minlength=4096).to(torch.float64)
        if counts.sum().item() != len(paths) * 32:
            raise ValueError("TiTok token count does not match 32 tokens per image")
        return (counts / counts.sum()).cpu().tolist()

    result = {"ok": True, "real": histogram(request["reference_paths"]),
              "generated": histogram(request["generated_paths"]), "device": device_name}
except Exception as exc:
    result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
Path(args.output).write_text(json.dumps(result), encoding="utf-8")
"""


class CHDModule(PipelineModule):
    """Released marginal-histogram CHD-1D for two independent image sets."""

    name = "chd"
    description = "Released TiTok-L-32 unigram CHD-1D distribution distance"
    input_type = "image"
    provenance = {"chd_1d": "adapted", "chd_metadata": "utility"}
    deviations = {
        "chd_1d": (
            "Uses the released CHD-1D unigram algorithm and public ImageNet TiTok-L-32. "
            "It does not claim the paper's CHD-1D/CHD-2D composite or retrained tokenizer. "
            "Ayase adds strict set validation and isolated pinned asset provisioning."
        ),
        "chd_metadata": "Ayase-added runtime trace; not an upstream CHD scalar.",
    }
    sources = {
        "chd_1d": "Released CHD-1D source at https://github.com/zexiJia/1d-Distance/tree/"
        + SOURCE_REVISION,
        "chd_metadata": "Ayase runtime trace for the released CHD-1D integration",
    }
    implementation_sources = (
        "../third_party/chd/LICENSE",
        "../third_party/chd/NOTICE",
        "../managed_runtime.py",
    )
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "revision": MODEL_REVISION,
            "task": "ImageNet TiTok-L-32 tokenizer for released CHD-1D",
            "size": "2,564,295,300 bytes",
            "auto_download": True,
            "notes": "Apache-2.0; model.safetensors SHA256 " + MODEL_SHA256,
        }
    ]
    metric_info = {
        "chd_1d": "Released unigram codebook Hellinger distance (0-1, lower=better)",
        "chd_metadata": "Structured CHD protocol, asset identity, device, and set counts",
    }
    default_config = {
        "reference_paths": None,
        "reference_dir": None,
        "source_root": None,
        "model_path": None,
        "models_dir": "models",
        "auto_download": True,
        "backend_python": None,
        "device": "auto",
        "batch_size": 128,
        "timeout": 7200,
    }
    required_packages = []

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._source_root: Optional[Path] = None
        self._model_root: Optional[Path] = None
        self._managed_python: Optional[Path] = None
        self._generated_paths: List[Path] = []
        self._invalid_generated = False
        self.last_metadata = {}

    def _resolve_python(self) -> Path:
        override = self.config.get("backend_python")
        if override:
            path = Path(override).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"CHD backend_python is not a file: {path}")
            return path
        if self._managed_python is None:
            # PipelineModule's generic managed bridge provisions CHD's registered
            # backend dependencies before this module runs. A subprocess is still
            # required to isolate upstream's generic top-level ``modeling`` package.
            self._managed_python = Path(sys.executable).resolve()
        return self._managed_python

    def _resolve_assets(self) -> tuple[Path, Path]:
        models_dir = Path(self.config.get("models_dir", "models")).expanduser().resolve()
        source_override = self.config.get("source_root")
        managed_source = models_dir / ".ayase" / "sources" / f"chd-{SOURCE_REVISION}"
        if source_override:
            source = Path(source_override).expanduser().resolve()
        elif managed_source.is_dir():
            source = managed_source
        elif self.config.get("auto_download", True):
            source = _ensure_source(managed_source)
        else:
            raise FileNotFoundError("CHD source is not provisioned")
        _verify_source(source, managed=not bool(source_override))
        model_override = self.config.get("model_path")
        if model_override:
            model = Path(model_override).expanduser().resolve()
        elif self.config.get("auto_download", True):
            model = download_hf_snapshot(
                MODEL_ID,
                str(models_dir),
                revision=MODEL_REVISION,
                allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE*"],
            )
        else:
            raise FileNotFoundError("CHD tokenizer is not provisioned")
        for filename in ("config.json", "model.safetensors"):
            if not (model / filename).is_file():
                raise FileNotFoundError(f"Incomplete CHD tokenizer snapshot: missing {filename}")
        digest = hashlib.sha256()
        with (model / "model.safetensors").open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("CHD model.safetensors digest mismatch")
        return source, model

    def setup(self) -> None:
        if self.test_mode:
            return
        self._ml_available = False
        self._backend = "unavailable"
        try:
            if int(self.config.get("batch_size", 128)) < 1:
                raise ValueError("CHD batch_size must be positive")
            if self._references() is None:
                raise ValueError("CHD requires a non-empty valid reference image set")
            self._resolve_python()
            self._source_root, self._model_root = self._resolve_assets()
            self._ml_available = True
            self._backend = f"chd-1d:titok-l32@{MODEL_REVISION}"
        except Exception as exc:
            logger.warning("CHD unavailable: %s", exc)

    @staticmethod
    def _valid_image(path: Path) -> bool:
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.is_file():
            return False
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, ValueError):
            return False

    def process(self, sample: Sample) -> Sample:
        if sample.is_video:
            return sample
        try:
            path = Path(sample.path).expanduser().resolve()
            if self._valid_image(path):
                self._generated_paths.append(path)
                return sample
        except (OSError, TypeError, ValueError):
            path = sample.path
        self._invalid_generated = True
        logger.warning("CHD rejected invalid generated image: %s", path)
        return sample

    def _references(self) -> Optional[List[Path]]:
        configured = self.config.get("reference_paths")
        directory = self.config.get("reference_dir")
        if configured is not None and directory is not None:
            logger.warning("CHD accepts reference_paths or reference_dir, not both")
            return None
        if configured is not None:
            if isinstance(configured, (str, bytes, os.PathLike)):
                logger.warning("CHD reference_paths must be a collection, not a scalar path")
                return None
            try:
                paths = [Path(value).expanduser().resolve() for value in configured]
            except (OSError, TypeError, ValueError):
                logger.warning("CHD reference_paths must be a collection of image paths")
                return None
        elif directory is not None:
            root = Path(directory).expanduser().resolve()
            if not root.is_dir():
                logger.warning("CHD reference_dir is not a directory: %s", root)
                return None
            paths = sorted(
                (
                    path.resolve()
                    for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                ),
                key=str,
            )
        else:
            logger.warning("CHD requires an independent reference_paths or reference_dir set")
            return None
        if not paths or any(not self._valid_image(path) for path in paths):
            logger.warning("CHD rejected an empty or invalid reference image set")
            return None
        return paths

    def _invoke(self, generated: Sequence[Path], references: Sequence[Path]) -> dict:
        if self._source_root is None or self._model_root is None:
            raise RuntimeError("CHD assets are unavailable")
        with tempfile.TemporaryDirectory(prefix="ayase-chd-") as name:
            root = Path(name)
            request = root / "request.json"
            output = root / "result.json"
            worker = root / "worker.py"
            request.write_text(
                json.dumps(
                    {
                        "source_root": str(self._source_root),
                        "model_root": str(self._model_root),
                        "generated_paths": [str(path) for path in generated],
                        "reference_paths": [str(path) for path in references],
                        "batch_size": int(self.config.get("batch_size", 128)),
                        "device": str(self.config.get("device", "auto")),
                    }
                ),
                encoding="utf-8",
            )
            worker.write_text(_WORKER, encoding="utf-8")
            result = subprocess.run(
                [
                    str(self._resolve_python()),
                    str(worker),
                    "--input",
                    str(request),
                    "--output",
                    str(output),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
                timeout=float(self.config.get("timeout", 7200)),
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                env=os.environ.copy(),
            )
            if not output.is_file():
                raise RuntimeError(f"CHD worker exited {result.returncode} without a result")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if result.returncode != 0 or not payload.get("ok"):
                raise RuntimeError(payload.get("error", "CHD worker failed"))
            return payload

    def post_process(self, all_samples: List[Sample]) -> None:
        del all_samples
        generated = list(self._generated_paths)
        references = self._references()
        if (
            not self._ml_available
            or self._invalid_generated
            or not generated
            or references is None
            or self.pipeline is None
        ):
            return
        try:
            payload = self._invoke(generated, references)
            score = _hellinger_distance(payload["real"], payload["generated"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("CHD worker returned an invalid histogram distance")
            source_override = bool(self.config.get("source_root"))
            model_override = bool(self.config.get("model_path"))
            marker = self._source_root / ".ayase-source-revision" if self._source_root else None
            source_revision = (
                marker.read_text(encoding="ascii").strip()
                if marker is not None and marker.is_file()
                else None
            )
            self.last_metadata = {
                "generated_count": len(generated),
                "reference_count": len(references),
                "source": {
                    "path": str(self._source_root),
                    "revision": source_revision,
                    "expert_override": source_override,
                },
                "model": {
                    "path": str(self._model_root),
                    "id": None if model_override else MODEL_ID,
                    "revision": None if model_override else MODEL_REVISION,
                    "safetensors_sha256": MODEL_SHA256,
                    "expert_override": model_override,
                },
                "device": payload.get("device"),
                "protocol": "direct RGB 256x256 /255; 32 tokens; 4096-bin unigram Hellinger",
            }
            self.pipeline.add_dataset_metric("chd_1d", score)
            self.pipeline.add_dataset_metric("chd_metadata", self.last_metadata)
        except Exception as exc:
            logger.warning("CHD computation failed: %s", exc)

    def on_execute(self) -> None:
        self._generated_paths.clear()
        self._invalid_generated = False
        self.last_metadata = {}

    def on_dispose(self) -> None:
        self._generated_paths.clear()
        self._invalid_generated = False
        self._ml_available = False
        self._backend = None
        super().on_dispose()


__all__ = ["CHDModule"]
