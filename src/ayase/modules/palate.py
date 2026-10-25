"""Original DINOv2 PALATE memorization and holistic distribution metrics.

Lower M_PALATE is better. This uses the released std-only unnormalized KDE
convention (not the paper's normalized Scott KDE) and non-L2-normalized CLS features.
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
from typing import List, Optional, Sequence

import numpy as np
from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)
PALATE_REVISION = "34568dd9c5bc8952fb9ae70e334ae57b485aedb0"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_ARCHIVE = "https://codeload.github.com/facebookresearch/dinov2/zip/" + DINOV2_REVISION
MODEL_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth?versionId=h4N1HKx3dh9OZaDGSn.oQQ2aSCK9RZFa"
MODEL_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
MODEL_SIZE = 1_217_586_395
_IMAGES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_REQUIRED = ("hubconf.py", "dinov2/models/vision_transformer.py", "LICENSE")


def _kernel_mean(x: np.ndarray, y: np.ndarray, sigma: float, block: int) -> float:
    total = 0.0
    anchor = np.asarray(x[0], dtype=np.float64)
    for i in range(0, len(x), block):
        a = np.asarray(x[i : i + block], dtype=np.float64) - anchor
        a2 = np.einsum("ij,ij->i", a, a)
        for j in range(0, len(y), block):
            b = np.asarray(y[j : j + block], dtype=np.float64) - anchor
            d2 = np.maximum(a2[:, None] + np.einsum("ij,ij->i", b, b)[None] - 2 * a @ b.T, 0)
            total += float(np.exp(-d2 / (2 * sigma * sigma)).sum(dtype=np.float64))
    return total / (len(x) * len(y))


def _mmd(
    x: np.ndarray, y: np.ndarray, sigma: float, block: int
) -> tuple[float, float, float, float]:
    xx, yy, xy = (
        _kernel_mean(x, x, sigma, block),
        _kernel_mean(y, y, sigma, block),
        _kernel_mean(x, y, sigma, block),
    )
    value = xx + yy - 2 * xy
    if -1e-12 < value < 0:
        value = 0.0
    return value, xx, yy, xy


def _estimate_sigma(real: np.ndarray, block: int, root: Path) -> float:
    count = len(real) * (len(real) - 1) // 2
    if not count:
        raise ValueError("automatic sigma requires at least two real images")
    with tempfile.TemporaryDirectory(prefix="palate-pairs-", dir=root) as name:
        path = Path(name) / "pairwise.float64"
        values = np.memmap(path, dtype=np.float64, mode="w+", shape=(count,))
        cursor = 0
        try:
            for i, row in enumerate(real[:-1]):
                for j in range(i + 1, len(real), block):
                    delta = np.asarray(real[j : j + block], dtype=np.float64) - row
                    part = np.einsum("ij,ij->i", delta, delta)
                    values[cursor : cursor + len(part)] = part
                    cursor += len(part)
            values.flush()
            lower, upper = (count - 1) // 2, count // 2
            values.partition((lower, upper))
            median = (float(values[lower]) + float(values[upper])) / 2.0
        finally:
            del values
    sigma = math.sqrt(max(0.0, median)) / 2
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("automatic sigma is degenerate")
    return sigma


def _kde(
    real: np.ndarray, generated: np.ndarray, threshold: float, block: int
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.std(real, axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("source KDE is degenerate: zero-variance real feature")
    output = np.empty(len(generated), dtype=np.float64)
    center = np.mean(real, axis=0)
    for start in range(0, len(generated), block):
        q = generated[start : start + block]
        maximum = np.full(len(q), -np.inf)
        accum = np.zeros(len(q))
        for offset in range(0, len(real), block):
            query = (q - center) / scale
            data = (real[offset : offset + block] - center) / scale
            squared = np.maximum(
                np.einsum("ij,ij->i", query, query)[:, None]
                + np.einsum("ij,ij->i", data, data)[None]
                - 2.0 * query @ data.T,
                0.0,
            )
            terms = -0.5 * squared
            updated = np.maximum(maximum, terms.max(1))
            accum = accum * np.exp(maximum - updated) + np.exp(terms - updated[:, None]).sum(1)
            maximum = updated
        output[start : start + len(q)] = maximum + np.log(accum) - math.log(len(real))
    return output >= threshold, output


def _compute_scores(
    train, test, generated, *, sigma=None, threshold=-300.0, block_size=1024, temporary_dir=None
) -> dict:
    train, test, generated = (np.asarray(v, dtype=np.float64) for v in (train, test, generated))
    if (
        any(
            v.ndim != 2 or not len(v) or not v.shape[1] or np.any(~np.isfinite(v))
            for v in (train, test, generated)
        )
        or len({v.shape[1] for v in (train, test, generated)}) != 1
    ):
        raise ValueError("PALATE requires three non-empty finite 2D arrays of equal width")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    real = np.concatenate((train, test))
    if sigma is None:
        root = Path(temporary_dir or tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="palate-sigma-", dir=root) as name:
            sigma = _estimate_sigma(real, block_size, Path(name))
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    base_mmd, ktt, kgg, ktg = _mmd(test, generated, sigma, block_size)
    base_den = ktt + kgg
    baseline = base_mmd / base_den if base_den > 0 else None
    try:
        mask, densities = _kde(real, generated, float(threshold), block_size)
    except ValueError as exc:
        return {
            "sigma": sigma,
            "threshold": float(threshold),
            "retained_count": None,
            "generated_count": len(generated),
            "retained_fraction": None,
            "baseline_score": baseline,
            "baseline_mmd": base_mmd,
            "baseline_ktt": ktt,
            "baseline_kgg": kgg,
            "baseline_ktg": ktg,
            "log_density_min": None,
            "log_density_max": None,
            "palate_score": None,
            "m_palate_score": None,
            "train_retained_mmd": None,
            "test_retained_mmd": None,
            "filter_error": str(exc),
        }
    retained = generated[mask]
    result = {
        "sigma": sigma,
        "threshold": float(threshold),
        "retained_count": len(retained),
        "generated_count": len(generated),
        "retained_fraction": len(retained) / len(generated),
        "baseline_score": baseline,
        "baseline_mmd": base_mmd,
        "baseline_ktt": ktt,
        "baseline_kgg": kgg,
        "baseline_ktg": ktg,
        "log_density_min": float(densities.min()),
        "log_density_max": float(densities.max()),
        "palate_score": None,
        "m_palate_score": None,
        "train_retained_mmd": None,
        "test_retained_mmd": None,
    }
    if not len(retained):
        return result
    train_mmd = _mmd(train, retained, sigma / 3, block_size)[0]
    test_mmd = _mmd(test, retained, sigma / 3, block_size)[0]
    result.update(train_retained_mmd=train_mmd, test_retained_mmd=test_mmd)
    denominator = train_mmd + test_mmd
    if denominator > 0 and baseline is not None:
        result["palate_score"] = test_mmd / denominator
        result["m_palate_score"] = 0.5 * baseline + 0.5 * result["palate_score"]
    return result


def _verify_source(root: Path, managed: bool) -> None:
    if any(not (root / p).is_file() for p in _REQUIRED):
        raise ValueError("incomplete DINOv2 source")
    marker = root / ".ayase-source-revision"
    if managed and not marker.is_file():
        raise ValueError("unverified DINOv2 source")
    if marker.is_file() and marker.read_text(encoding="ascii").strip() != DINOV2_REVISION:
        raise ValueError("DINOv2 revision mismatch")


def _ensure_source(destination: Path) -> Path:
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if destination.is_dir():
            _verify_source(destination, True)
            return destination
        with tempfile.TemporaryDirectory(prefix="palate-source-", dir=destination.parent) as name:
            archive = Path(name) / "source.zip"
            urllib.request.urlretrieve(DINOV2_ARCHIVE, archive)
            unpacked = Path(name) / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                roots = {Path(x.filename).parts[0] for x in bundle.infolist() if x.filename}
                if len(roots) != 1:
                    raise ValueError("invalid DINOv2 archive")
                base = unpacked.resolve()
                for item in bundle.infolist():
                    target = (unpacked / item.filename).resolve()
                    if target != base and base not in target.parents:
                        raise ValueError("unsafe DINOv2 archive")
                    if item.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError("DINOv2 archive contains symlink")
                bundle.extractall(unpacked)
            extracted = unpacked / roots.pop()
            (extracted / ".ayase-source-revision").write_text(DINOV2_REVISION, encoding="ascii")
            _verify_source(extracted, True)
            extracted.replace(destination)
    return destination


def _download_model(destination: Path) -> Path:
    from ayase.managed_runtime import _SetupLock

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _SetupLock(destination.with_suffix(".lock")):
        if not destination.is_file():
            part = destination.with_suffix(".part")
            try:
                urllib.request.urlretrieve(MODEL_URL, part)
                part.replace(destination)
            finally:
                part.unlink(missing_ok=True)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(data)
        if destination.stat().st_size != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
            raise ValueError("DINOv2 checkpoint identity mismatch")
    return destination


_WORKER = r"""import json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); out = {"ok": False}
try:
    source = Path(r["source_root"]).resolve(); os.chdir(source); sys.path.insert(0, str(source))
    with torch.device("meta"):
        model = torch.hub.load(str(source), "dinov2_vitl14", source="local", pretrained=False)
    state = torch.load(r["model_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    device = r["device"]
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().requires_grad_(False).to(device)
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1); std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    def embed(paths):
        result = []
        with torch.no_grad():
            for start in range(0, len(paths), r["batch_size"]):
                images = []
                for path in paths[start:start + r["batch_size"]]:
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB").resize((224, 224), bicubic), dtype=np.float32).copy()
                    images.append((torch.from_numpy(array).permute(2, 0, 1) / 255 - mean) / std)
                result.append(model(torch.stack(images).to(device)).cpu().numpy())
        return np.concatenate(result)
    np.savez(sys.argv[3], train=embed(r["train"]), test=embed(r["test"]), generated=embed(r["generated"]))
    out = {"ok": True, "device": device}
except Exception as exc: out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
Path(sys.argv[2]).write_text(json.dumps(out), encoding="utf-8")"""


class PALATEModule(PipelineModule):
    name = "palate"
    description = "Original DINOv2 PALATE and M_PALATE distribution metrics"
    input_type = "image"
    provenance = {
        k: "adapted"
        for k in (
            "palate_score",
            "m_palate_score",
            "palate_baseline_score",
            "palate_retained_fraction",
        )
    } | {"palate_metadata": "utility"}
    deviations = {
        key: (
            "Original DINOv2 and released std-only unnormalized KDE, not DINOv3/normalized "
            "paper KDE; Ayase adds exact strict-upper real-pair median sigma estimation and "
            "float64 blockwise reductions."
        )
        for key in (
            "palate_score",
            "m_palate_score",
            "palate_baseline_score",
            "palate_retained_fraction",
        )
    } | {"palate_metadata": "Ayase runtime diagnostics."}
    sources = {
        key: "https://github.com/gmum/PALATE/tree/" + PALATE_REVISION
        for key in provenance
        if key != "palate_metadata"
    } | {"palate_metadata": "https://arxiv.org/abs/2503.18462"}
    implementation_sources = ("../third_party/palate/LICENSE", "../third_party/palate/NOTICE")
    models = [
        {
            "id": "facebookresearch/dinov2:dinov2_vitl14",
            "type": "torch_hub",
            "revision": DINOV2_REVISION,
            "url": MODEL_URL,
            "task": "PALATE features",
            "size": "1,217,586,395 bytes",
            "vram": "~1.22 GB weights plus inference activations (not benchmarked)",
            "auto_download": True,
            "notes": "Apache-2.0; SHA256 " + MODEL_SHA256,
        }
    ]
    metric_info = {
        "palate_score": "Training-vs-heldout memorization ratio (balanced generalization near 0.5)",
        "m_palate_score": "Holistic score (lower=better)",
        "palate_baseline_score": "Scaled full-set MMD baseline",
        "palate_retained_fraction": "KDE-retained generated fraction",
        "palate_metadata": "Protocol and diagnostics",
    }
    default_config = {
        "train_paths": None,
        "train_dir": None,
        "test_paths": None,
        "test_dir": None,
        "sigma": None,
        "kde_threshold": -300.0,
        "block_size": 1024,
        "batch_size": 64,
        "source_root": None,
        "model_path": None,
        "models_dir": "models",
        "auto_download": True,
        "backend_python": None,
        "device": "auto",
        "timeout": 7200,
    }
    required_packages = []

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._generated_paths = []
        self._invalid_generated = False
        self._ml_available = False
        self._backend = None
        self.last_metadata = {}

    @staticmethod
    def _valid(path):
        if path.suffix.lower() not in _IMAGES or not path.is_file():
            return False
        try:
            from PIL import Image

            with Image.open(path) as im:
                im.verify()
            return True
        except (OSError, ValueError):
            return False

    def _paths(self, pk, dk):
        values, directory = self.config.get(pk), self.config.get(dk)
        if values is not None and directory is not None:
            return None
        if values is not None:
            if isinstance(values, (str, bytes, os.PathLike)):
                return None
            try:
                paths = [Path(v).expanduser().resolve() for v in values]
            except (OSError, TypeError, ValueError):
                return None
        elif directory is not None:
            root = Path(directory).expanduser().resolve()
            if not root.is_dir():
                return None
            paths = sorted(
                (
                    p.resolve()
                    for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMAGES
                ),
                key=str,
            )
        else:
            return None
        return (
            paths
            if paths and len(paths) == len(set(paths)) and all(self._valid(p) for p in paths)
            else None
        )

    def _real_sets(self):
        train, test = self._paths("train_paths", "train_dir"), self._paths("test_paths", "test_dir")
        if train is None or test is None or set(train) & set(test):
            logger.warning("PALATE requires valid disjoint train/test sets")
            return None
        return train, test

    def setup(self):
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        try:
            if self._real_sets() is None:
                return
            if int(self.config["block_size"]) < 1 or int(self.config["batch_size"]) < 1:
                raise ValueError("batch sizes must be positive")
            if self.config.get("sigma") is not None and (
                float(self.config["sigma"]) <= 0 or not math.isfinite(float(self.config["sigma"]))
            ):
                raise ValueError("sigma must be positive")
            self._ml_available = True
            self._backend = "palate:dinov2-vitl14"
        except Exception as exc:
            logger.warning("PALATE unavailable: %s", exc)

    def process(self, sample):
        if sample.is_video or getattr(sample, "reference_path", None) is not None:
            self._invalid_generated = True
            return sample
        try:
            path = Path(sample.path).expanduser().resolve()
        except (OSError, TypeError, ValueError):
            path = None
        if path is None or not self._valid(path) or path in self._generated_paths:
            self._invalid_generated = True
        else:
            self._generated_paths.append(path)
        return sample

    def _resolve_assets(self):
        root = Path(self.config["models_dir"]).expanduser().resolve()
        override = self.config.get("source_root")
        source = (
            Path(override).expanduser().resolve()
            if override
            else root / ".ayase" / "sources" / f"dinov2-{DINOV2_REVISION}"
        )
        if not source.is_dir():
            if override or not self.config["auto_download"]:
                raise FileNotFoundError("DINOv2 source unavailable")
            source = _ensure_source(source)
        _verify_source(source, not bool(override))
        model_override = self.config.get("model_path")
        model = (
            Path(model_override).expanduser().resolve()
            if model_override
            else root / ".ayase" / "models" / "dinov2_vitl14_pretrain.pth"
        )
        if not model.is_file() and (model_override or not self.config["auto_download"]):
            raise FileNotFoundError("DINOv2 checkpoint unavailable")
        if not model_override:
            model = _download_model(model)
        return source, model

    def _invoke(self, train, test, generated):
        source, model = self._resolve_assets()
        python = Path(self.config.get("backend_python") or sys.executable).resolve()
        if not python.is_file():
            raise FileNotFoundError("backend_python unavailable")
        tmp = Path(self.config["models_dir"]).resolve() / ".ayase" / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="palate-", dir=tmp) as name:
            root = Path(name)
            request = root / "request.json"
            output = root / "output.json"
            arrays = root / "features.npz"
            worker = root / "worker.py"
            request.write_text(
                json.dumps(
                    {
                        "source_root": str(source),
                        "model_path": str(model),
                        "train": list(map(str, train)),
                        "test": list(map(str, test)),
                        "generated": list(map(str, generated)),
                        "batch_size": int(self.config["batch_size"]),
                        "device": self.config["device"],
                    }
                ),
                encoding="utf-8",
            )
            worker.write_text(_WORKER, encoding="utf-8")
            done = subprocess.run(
                [str(python), str(worker), str(request), str(output), str(arrays)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=float(self.config["timeout"]),
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
            if not output.is_file():
                raise RuntimeError(f"worker exited {done.returncode}")
            payload = json.loads(output.read_text())
            if done.returncode or not payload.get("ok"):
                raise RuntimeError(payload.get("error", "worker failed"))
            if not arrays.is_file():
                raise RuntimeError("worker did not produce feature arrays")
            with np.load(arrays, allow_pickle=False) as loaded:
                payload.update(
                    {key: np.array(loaded[key]) for key in ("train", "test", "generated")}
                )
            payload.update(source_root=str(source), model_path=str(model))
            return payload

    def post_process(self, all_samples):
        del all_samples
        sets = self._real_sets()
        generated = list(self._generated_paths)
        if (
            not self._ml_available
            or self._invalid_generated
            or not generated
            or sets is None
            or self.pipeline is None
        ):
            return
        train, test = sets
        try:
            payload = self._invoke(train, test, generated)
            tmp = Path(self.config["models_dir"]).resolve() / ".ayase" / "tmp"
            tmp.mkdir(parents=True, exist_ok=True)
            scores = _compute_scores(
                payload["train"],
                payload["test"],
                payload["generated"],
                sigma=self.config.get("sigma"),
                threshold=float(self.config["kde_threshold"]),
                block_size=int(self.config["block_size"]),
                temporary_dir=tmp,
            )
            source_marker = Path(payload["source_root"]) / ".ayase-source-revision"
            source_revision = (
                source_marker.read_text(encoding="ascii").strip()
                if source_marker.is_file()
                else None
            )
            self.last_metadata = {
                **scores,
                "train_count": len(train),
                "test_count": len(test),
                "device": payload["device"],
                "source": {
                    "revision": source_revision,
                    "path": payload["source_root"],
                    "expert_override": bool(self.config.get("source_root")),
                },
                "model": {
                    "url": None if self.config.get("model_path") else MODEL_URL,
                    "sha256": MODEL_SHA256,
                    "path": payload["model_path"],
                    "expert_override": bool(self.config.get("model_path")),
                },
                "preprocessing": "PIL direct RGB 224x224 bicubic; ImageNet normalization",
                "features": "native ViT-L/14 final-LN CLS, not L2 normalized",
                "kde": "released std-only unnormalized logmeanexp, population std, >= threshold",
                "sigma_estimator": "explicit value or exact median of strict-upper real squared distances",
                "generated_train_path_overlap": len(set(generated) & set(train)),
                "generated_test_path_overlap": len(set(generated) & set(test)),
                "palate_source_revision": PALATE_REVISION,
            }
            for key, out in (
                ("baseline_score", "palate_baseline_score"),
                ("retained_fraction", "palate_retained_fraction"),
                ("palate_score", "palate_score"),
                ("m_palate_score", "m_palate_score"),
            ):
                if scores[key] is not None:
                    self.pipeline.add_dataset_metric(out, scores[key])
            self.pipeline.add_dataset_metric("palate_metadata", self.last_metadata)
        except Exception as exc:
            logger.warning("PALATE computation failed: %s", exc)

    def on_execute(self):
        self._generated_paths.clear()
        self._invalid_generated = False
        self.last_metadata = {}

    def on_dispose(self):
        self.on_execute()
        self._ml_available = False
        self._backend = None
        super().on_dispose()


__all__ = ["PALATEModule"]
