"""Standalone process worker for the dependency-isolated videojedi backend."""

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path


EXPECTED_VIDEJEDI_VERSION = "1.1.0"
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class FeatureCacheLock:
    """Cross-process lock protecting upstream's non-atomic feature writes."""

    def __init__(self, directory, timeout=300.0):
        self.path = Path(directory) / ".ayase-jedi.lock"
        self.timeout = float(timeout)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._lock()
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise TimeoutError(f"timed out waiting for JEDi cache lock: {self.path}")
                time.sleep(0.1)

    def _lock(self):
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is None:
            return None
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                self.handle.flush()
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
        return None


def validate_feature_cache(path, minimum_rows):
    """Validate one upstream NumPy feature file and close its mmap on Windows."""
    import numpy as np

    feature_path = Path(path)
    if not feature_path.is_file():
        return False
    array = None
    try:
        array = np.load(feature_path, allow_pickle=False, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] != int(minimum_rows) or array.shape[1] < 1:
            return False
        if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
            return False
        return True
    except Exception:
        return False
    finally:
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


def repair_feature_cache(directory, minimum_rows):
    """Remove only corrupt/incomplete derived feature arrays from an owned namespace."""
    root = Path(directory)
    status = {}
    for name in ("train.npy", "test.npy"):
        path = root / name
        valid = validate_feature_cache(path, minimum_rows)
        status[name] = valid
        if path.exists() and not valid:
            path.unlink()
            print(f"Removed invalid JEDi cache file: {path}", file=sys.stderr)
    return status


class VideoPathDataset:
    """Bounded PyAV loader matching Ayase's documented JEDi tensor contract."""

    def __init__(self, paths, num_frames=16, resolution=224):
        self.paths = [Path(path) for path in paths]
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self._decode(self.paths[index]), 0

    def _decode(self, path):
        import av
        import numpy as np
        import torch
        import torch.nn.functional as functional

        with av.open(str(path)) as container:
            count = sum(1 for _ in container.decode(video=0))
            if not count:
                raise ValueError(f"video contains no decodable frames: {path}")
            indices = np.linspace(0, count - 1, self.num_frames, dtype=np.int64)
            wanted = set(indices.tolist())
            container.seek(0)
            frames = {
                index: frame.to_ndarray(format="rgb24")
                for index, frame in enumerate(container.decode(video=0))
                if index in wanted
            }
        video = torch.from_numpy(np.stack([frames[index] for index in indices])).permute(
            0, 3, 1, 2
        )
        video = video.to(dtype=torch.float32).div_(255.0)
        return functional.interpolate(
            video,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )


def _package_status():
    try:
        installed = package_version("videojedi")
    except PackageNotFoundError:
        installed = None
    return installed, installed == EXPECTED_VIDEJEDI_VERSION


def check_backend():
    installed, supported = _package_status()
    package_spec = importlib.util.find_spec("videojedi")
    result = {
        "ok": False,
        "mode": "check",
        "videojedi_version": installed,
        "expected_videojedi_version": EXPECTED_VIDEJEDI_VERSION,
        "package_source": package_spec.origin if package_spec else None,
        "package_supported": supported,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "current_device": None,
        "required_visible_device": "cuda:0",
    }
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch

        result["torch_version"] = str(torch.__version__)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        if result["cuda_available"]:
            result["current_device"] = int(torch.cuda.current_device())
    except Exception as exc:
        result["error"] = f"torch check failed: {type(exc).__name__}: {exc}"
        return result
    result["ok"] = bool(
        supported
        and result["cuda_available"]
        and result["cuda_device_count"] >= 1
        and result["current_device"] == 0
    )
    if not result["ok"]:
        if not supported:
            result["error"] = (
                f"requires videojedi=={EXPECTED_VIDEJEDI_VERSION}, found {installed!r}"
            )
        elif not result["cuda_available"]:
            result["error"] = "videojedi requires CUDA"
        else:
            result["error"] = "videojedi requires the child-visible current CUDA device 0"
    return result


def _resolved_file(value, field):
    if not value:
        raise ValueError(f"{field} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} is not a file: {path}")
    return path


def _video_paths(values, field):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    paths = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"{field} contains an invalid video path: {path}")
        paths.append(path)
    return paths


def make_loader(paths, batch_size, num_workers, num_frames, resolution):
    from torch.utils.data import DataLoader

    return DataLoader(
        VideoPathDataset(paths, num_frames=num_frames, resolution=resolution),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def run_payload(payload):
    installed, supported = _package_status()
    if not supported:
        raise RuntimeError(
            f"requires videojedi=={EXPECTED_VIDEJEDI_VERSION}, found {installed!r}"
        )
    if payload.get("device", "cuda:0") != "cuda:0":
        raise ValueError("isolated JEDi worker supports only child-visible device cuda:0")
    num_frames = int(payload.get("num_frames", 16))
    resolution = int(payload.get("resolution", 224))
    if num_frames != 16 or resolution != 224:
        raise ValueError("JEDi worker requires num_frames=16 and resolution=224")
    generated = _video_paths(payload.get("generated_paths"), "generated_paths")
    references = _video_paths(payload.get("reference_paths"), "reference_paths")
    limit = min(int(payload.get("num_samples", 5000)), len(generated), len(references))
    if limit < 1:
        raise ValueError("num_samples and both input collections must be non-empty")
    generated = generated[:limit]
    references = references[:limit]
    model_dir = Path(payload.get("model_dir", "")).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir is not a directory: {model_dir}")
    for name in ("vith16.pth.tar", "ssv2-probe.pth.tar"):
        _resolved_file(model_dir / name, name)
    config_path = _resolved_file(payload.get("config_path"), "config_path")
    feature_path = Path(payload.get("feature_path", "")).expanduser().resolve()
    if not payload.get("feature_path"):
        raise ValueError("feature_path namespace is required")
    feature_path.mkdir(parents=True, exist_ok=True)
    cache_lock_timeout = float(payload.get("cache_lock_timeout", 300.0))
    if cache_lock_timeout <= 0:
        raise ValueError("cache_lock_timeout must be positive")
    batch_size = int(payload.get("batch_size", 4))
    num_workers = int(payload.get("num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from videojedi import JEDiMetric

    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("videojedi requires the child-visible current CUDA device 0")
    if torch.distributed.is_initialized():
        raise RuntimeError("worker started with an existing distributed process group")
    reference_loader = make_loader(
        references, batch_size, num_workers, num_frames, resolution
    )
    generated_loader = make_loader(
        generated, batch_size, num_workers, num_frames, resolution
    )
    with FeatureCacheLock(feature_path, timeout=cache_lock_timeout):
        repair_feature_cache(feature_path, limit)
        try:
            with contextlib.redirect_stdout(sys.stderr), torch.cuda.device("cuda:0"):
                metric = JEDiMetric(
                    feature_path=str(feature_path),
                    model_dir=str(model_dir),
                    config_path=str(config_path),
                )
                metric.load_features(
                    train_loader=reference_loader,
                    test_loader=generated_loader,
                    num_samples=limit,
                )
                if not all(repair_feature_cache(feature_path, limit).values()):
                    raise ValueError("videojedi produced an invalid or incomplete feature cache")
                score = float(metric.compute_metric())
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    if not math.isfinite(score):
        raise ValueError("videojedi returned a non-finite score")
    return {
        "ok": True,
        "mode": "score",
        "jedi_score": score,
        "generated_count": len(generated),
        "reference_count": len(references),
        "num_samples": limit,
        "videojedi_version": installed,
        "device": "cuda:0",
        "feature_path": str(feature_path),
    }


def _write_json(path, payload):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated official JEDi worker")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = check_backend()
        else:
            if not args.input:
                raise ValueError("--input is required unless --check is used")
            payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("input JSON must be an object")
            result = run_payload(payload)
        _write_json(args.output, result)
        return 0 if result.get("ok") else 1
    except Exception as exc:
        error = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            _write_json(args.output, error)
        except Exception as write_exc:
            print(f"failed to write worker result: {write_exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
