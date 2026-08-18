"""Compatibility shims for stale third-party dependencies.

This module is imported very early in ayase/__init__.py so that the patches
take effect before any consumer (pyiqa, ImageReward, RAM, etc.) imports the
problematic packages.

Patches:
    clip — OpenAI CLIP package (last released 2021) uses
           `from pkg_resources import packaging`, deprecated in modern
           setuptools. We pre-load our vendored copy (at ayase/_vendor/clip)
           into sys.modules['clip'] so any subsequent `import clip` resolves
           to the patched version.

    transformers.modeling_utils — older downstream libs (RAM bert.py,
           ImageReward BLIP med.py, etc.) still import
           `apply_chunking_to_forward`, `find_pruneable_heads_and_indices`,
           and `prune_linear_layer` from `transformers.modeling_utils`,
           but newer transformers (>=4.31?) moved them to
           `transformers.pytorch_utils`. We re-export them on the old path
           so those libs import without source patches.

This module is idempotent: re-importing or repeated invocation is a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


_PATCHED_FLAG = "_ayase_compat_applied"
_VENDOR_DIR = Path(__file__).parent / "_vendor"


def _install_vendored_package(name: str, vendor_subdir: str) -> None:
    """Register a vendored package in sys.modules under the given name.

    Subsequent `import {name}` anywhere in the process — including inside
    transitively-imported third-party libs — will resolve to the vendored copy
    instead of looking for {name} on the regular sys.path.
    """
    if name in sys.modules:
        return  # an already-loaded module wins; don't second-guess

    pkg_path = _VENDOR_DIR / vendor_subdir
    init_path = pkg_path / "__init__.py"
    if not init_path.exists():
        return  # vendor copy missing — silently skip; fall back to pip-installed

    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(pkg_path)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _backport_modeling_utils() -> None:
    """Re-export symbols that moved from modeling_utils → pytorch_utils.

    Old libs (RAM bert.py, ImageReward BLIP med.py, etc.) still do
    `from transformers.modeling_utils import apply_chunking_to_forward, ...`
    which fails on newer transformers. We add the symbols back to the
    modeling_utils namespace so the legacy import path resolves.
    """
    try:
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
    except ImportError:
        return
    for sym in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(_mu, sym) and hasattr(_pu, sym):
            setattr(_mu, sym, getattr(_pu, sym))


def apply_patches() -> None:
    """Apply all compatibility patches. Safe to call multiple times."""
    if getattr(sys, _PATCHED_FLAG, False):
        return

    # OpenAI CLIP: vendored copy bypasses deprecated pkg_resources.packaging.
    _install_vendored_package("clip", "clip")

    # transformers.modeling_utils → pytorch_utils symbol relocations.
    _backport_modeling_utils()

    setattr(sys, _PATCHED_FLAG, True)


_PADDLE_GPU_INDEXES = {
    "11.8": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    "12.0": "https://www.paddlepaddle.org.cn/packages/stable/cu120/",
    "12.3": "https://www.paddlepaddle.org.cn/packages/stable/cu123/",
    "12.6": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
}
_PADDLE_GPU_VERSION = "3.3.1"
_PADDLE_GPU_ENSURED = False
_PADDLE_LOCK_STALE_SECONDS = 3600


def _paddle_lock_path() -> Path:
    """Lock file beside the installed package, i.e. one lock per environment."""
    return Path(__file__).resolve().parent.parent / ".ayase_paddle_gpu.lock"


def _paddle_gpu_installed() -> bool:
    import importlib.metadata as _md
    try:
        _md.distribution("paddlepaddle-gpu")
        return True
    except _md.PackageNotFoundError:
        return False


def _verify_paddle_gpu(python: str) -> bool:
    """Run a real GPU op in a fresh interpreter.

    Probing in this process would prove nothing: paddle may already be imported
    here, and a freshly installed wheel is not picked up by a running process.
    """
    import subprocess
    probe = (
        "import paddle;"
        "assert paddle.device.is_compiled_with_cuda();"
        "assert paddle.device.cuda.device_count() > 0;"
        "x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], place=paddle.CUDAPlace(0));"
        "assert 'gpu' in str((x @ x).place)"
    )
    try:
        return subprocess.run(
            [python, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        ).returncode == 0
    except Exception:
        return False


def ensure_paddle_gpu() -> None:
    """Swap CPU paddlepaddle for paddlepaddle-gpu when CUDA is available.

    PaddlePaddle 3.x publishes GPU wheels only on paddle.org's simple index,
    not standard PyPI. ``pip install ayase`` therefore lands a CPU paddle.
    On a CUDA host we install the matching GPU build at runtime so OCR modules
    just work.

    The install runs with ``--no-deps`` deliberately. The GPU wheel pins fifteen
    ``nvidia-*-cu12`` distributions with ``==`` at the CUDA line it was built
    against, and torch pins those same fifteen distributions, also with ``==``,
    at its own line. No resolution satisfies both, so whoever installs last
    wins — and running from an import, that is always paddle. The damage is
    silent: torch keeps working, merely against older CUDA libraries than it
    was built for, and nothing ever reports it.

    Dropping the dependencies is safe rather than merely expedient:

    * torch is a hard dependency of ayase, so its CUDA libraries are present in
      every environment this code can run in, and CUDA minor releases are
      backward compatible — a cu126 paddle build runs on cu128 libraries;
    * the GPU wheel's non-CUDA requirements are identical to those of the CPU
      wheel, which is itself a hard dependency and therefore already installed.

    Several processes may import ayase at once, so the install is serialised by
    an exclusive lock file: latecomers wait, then find the package installed
    and return instead of running a second pip into the same environment.

    No-op on non-CUDA hosts, when paddle is already GPU-compiled, when the CUDA
    version is unsupported, or when pip is unavailable.
    """
    global _PADDLE_GPU_ENSURED
    if _PADDLE_GPU_ENSURED:
        return
    _PADDLE_GPU_ENSURED = True

    # Probe via metadata only — do NOT `import paddle` here, otherwise the CPU
    # build ends up cached in sys.modules and pip-install of the GPU build can't
    # take effect within the same process.
    import importlib.metadata as _md
    if _paddle_gpu_installed():
        return  # GPU paddle already installed
    try:
        _md.distribution("paddlepaddle")
    except _md.PackageNotFoundError:
        return  # no paddle at all; nothing to swap

    cuda = None
    try:
        import torch  # type: ignore[import-not-found]
        cuda = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    if not cuda:
        return

    cuda_key = ".".join(cuda.split(".")[:2])
    if cuda_key not in _PADDLE_GPU_INDEXES:
        target_major = cuda.split(".")[0]
        candidates = [k for k in _PADDLE_GPU_INDEXES if k.split(".")[0] == target_major]
        if not candidates:
            return
        cuda_key = max(candidates, key=lambda v: tuple(int(x) for x in v.split(".")))

    index = _PADDLE_GPU_INDEXES[cuda_key]

    import logging
    import os
    import subprocess
    import time

    log = logging.getLogger("ayase._compat")
    lock = _paddle_lock_path()

    try:
        if lock.exists() and time.time() - lock.stat().st_mtime > _PADDLE_LOCK_STALE_SECONDS:
            lock.unlink()  # holder died mid-install and is not coming back
    except OSError:
        pass

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.time() + _PADDLE_LOCK_STALE_SECONDS
        while time.time() < deadline:
            if _paddle_gpu_installed():
                return
            if not lock.exists():
                return
            time.sleep(2)
        return
    except OSError as e:
        log.warning("paddlepaddle-gpu auto-install skipped, cannot take lock: %s", e)
        return

    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        log.warning("Installing paddlepaddle-gpu==%s for CUDA %s (one-time, ~600MB)",
                    _PADDLE_GPU_VERSION, cuda_key)
        cmd = [
            sys.executable, "-m", "pip", "install", "--no-deps",
            f"paddlepaddle-gpu=={_PADDLE_GPU_VERSION}",
            "--index-url", index,
        ]
        subprocess.check_call(cmd)
        if not _verify_paddle_gpu(sys.executable):
            log.warning(
                "paddlepaddle-gpu==%s installed but no GPU op succeeded; "
                "paddle modules fall back to CPU",
                _PADDLE_GPU_VERSION,
            )
    except Exception as e:
        log.warning("paddlepaddle-gpu auto-install failed: %s", e)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


apply_patches()
