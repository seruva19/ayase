"""Bootstrap helpers — auto-install platform-specific deps that pip cannot
resolve from a pyproject.toml (notably paddlepaddle-gpu, which is published
only via paddle.org's own simple index, not standard PyPI).

Usage::

    ayase-bootstrap --gpu                  # detect CUDA, install paddle-gpu
    ayase-bootstrap --gpu --cuda 12.6      # force CUDA version
    ayase-bootstrap --gpu --dry-run        # show what would be done
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys


PADDLE_GPU_INDEXES: dict[str, str] = {
    "11.8": "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    "12.0": "https://www.paddlepaddle.org.cn/packages/stable/cu120/",
    "12.3": "https://www.paddlepaddle.org.cn/packages/stable/cu123/",
    "12.6": "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
}

DEFAULT_PADDLE_VERSION = "3.3.1"


def _cuda_from_torch() -> str | None:
    try:
        import torch
        v = getattr(torch.version, "cuda", None)
        return v if isinstance(v, str) and v else None
    except Exception:
        return None


def _cuda_from_nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    return m.group(1) if m else None


def _detect_cuda() -> str | None:
    return _cuda_from_torch() or _cuda_from_nvidia_smi()


def _pick_supported(cuda: str) -> str | None:
    if cuda in PADDLE_GPU_INDEXES:
        return cuda
    major_minor = ".".join(cuda.split(".")[:2])
    if major_minor in PADDLE_GPU_INDEXES:
        return major_minor
    target_major = int(cuda.split(".")[0])
    candidates = [k for k in PADDLE_GPU_INDEXES if int(k.split(".")[0]) == target_major]
    if not candidates:
        return None
    return max(candidates, key=lambda v: tuple(int(x) for x in v.split(".")))


def install_paddle_gpu(cuda: str | None = None, paddle_version: str = DEFAULT_PADDLE_VERSION,
                       dry_run: bool = False) -> int:
    cuda = cuda or _detect_cuda()
    if not cuda:
        print("Could not detect CUDA. Pass --cuda <version>, or install paddlepaddle-gpu manually.",
              file=sys.stderr)
        return 1
    supported = _pick_supported(cuda)
    if not supported:
        print(f"No paddlepaddle-gpu wheel available for CUDA {cuda}. "
              f"Supported: {sorted(PADDLE_GPU_INDEXES)}", file=sys.stderr)
        return 2

    index = PADDLE_GPU_INDEXES[supported]
    cmd = [sys.executable, "-m", "pip", "install", f"paddlepaddle-gpu=={paddle_version}",
           "--index-url", index]
    print(f"[ayase-bootstrap] detected CUDA {cuda} (using paddle wheel for cu{supported.replace('.', '')})")
    print(f"[ayase-bootstrap] {' '.join(cmd)}")
    if dry_run:
        return 0

    if shutil.which("pip"):
        return subprocess.call(cmd)
    print("pip not on PATH; cannot install.", file=sys.stderr)
    return 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ayase-bootstrap",
                                description="Install platform-specific Ayase deps.")
    p.add_argument("--gpu", action="store_true",
                   help="Install paddlepaddle-gpu for GPU OCR (paddleocr 3.x).")
    p.add_argument("--cuda", help="Force CUDA version (e.g., 12.6). Otherwise auto-detected.")
    p.add_argument("--paddle-version", default=DEFAULT_PADDLE_VERSION,
                   help=f"paddlepaddle-gpu version (default {DEFAULT_PADDLE_VERSION}).")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen, don't install.")
    args = p.parse_args(argv)

    if args.gpu:
        return install_paddle_gpu(cuda=args.cuda, paddle_version=args.paddle_version,
                                  dry_run=args.dry_run)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
