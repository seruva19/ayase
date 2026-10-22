"""Managed source, environments, and checkpoints for T2V-CompBench."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ayase.managed_runtime import _SetupLock, ensure_managed_environment

REVISION = "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda"
SOURCE_URL = f"https://codeload.github.com/KaiyueSun98/T2V-CompBench/zip/{REVISION}"

# Git object identities from the pinned upstream commit.  Verifying these files
# makes a corrupt or substituted archive unusable without redistributing it.
PINNED_BLOBS: Mapping[str, str] = {
    "Depth-Anything/compbench_run_depth.py": "fcb535f4850edb644bffde0f0e2c8a877ca29d31",
    "Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py": "2bf9f668ab81994812c18e4cefa83ef54f435abc",
    "Grounded-Segment-Anything/compbench_eval_spatial_relationships.py": "a6d71fa61a36bbd783d40e07e443fb0c2c81d365",
    "Grounded-Segment-Anything/compbench_motion_binding_seg.py": "4a8caa2233c1227daf049617ae14b8f0fd01b7bc",
    "LLaVA/llava/eval/compbench_eval_action_binding.py": "4e3f38167877b88144a286b7c75b9a0c9913c4e9",
    "LLaVA/llava/eval/compbench_eval_consistent_attr.py": "30006a67f435df46b47e9ecbcf6c2cb84e038360",
    "LLaVA/llava/eval/compbench_eval_dynamic_attr.py": "6764278b57972cee21d0558a21a1f646786ab894",
    "LLaVA/llava/eval/compbench_eval_interaction.py": "f824ea47c2386d78aed9271a0c0ed92507f6b9dc",
    "dot/compbench_eval_motion_binding.py": "7214a3db689bd72a3b58da88ec21d20dbc76deb8",
    "dot/dot/utils/options/compbench_demo_options.py": "9bc5172f83f04d1ed896d3dada6ee30646114af1",
    **{
        f"meta_data/{name}.json": digest
        for name, digest in {
            "action_binding": "515db9bac3deffb1264700e2b7003c74e21a8b99",
            "consistent_attribute_binding": "1efc395dd4e608842fc1a75046b217eefe1cb94e",
            "dynamic_attribute_binding": "6459bda1d00a19ab9e1655eaf82cd136ee29bdfd",
            "generative_numeracy": "e33e52c017a996f6aedf735d213eea10f8c924f5",
            "motion_binding": "2fc50f83876decf33aa50404937474830032b527",
            "object_interactions": "440140890487a219429da8c873ecf068e5aa4b17",
            "spatial_relationships": "a6a90d4010ebff540b494af42feda87c3b2f8ea7",
        }.items()
    },
}

ASSETS: Mapping[str, Tuple[str, str]] = {
    "Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth": (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "groundingdino",
    ),
    "Grounded-Segment-Anything/sam_vit_h_4b8939.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam"
    ),
    "dot/checkpoints/cvo_raft_patch_8.pth": (
        "https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth", "dot"
    ),
    "dot/checkpoints/movi_f_raft_patch_4_alpha.pth": (
        "https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth", "dot"
    ),
    "dot/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth": (
        "https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth", "dot"
    ),
}

LLAVA_REQUIREMENTS = ("pip>=24.0",)
DETECTION_REQUIREMENTS = (
    "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2",
    "transformers==4.40.2", "tokenizers==0.19.1", "timm==0.9.16",
    "numpy==1.26.4", "opencv-python==4.9.0.80", "pandas==2.2.2",
    "scipy==1.13.0", "matplotlib==3.8.4", "safetensors==0.4.3",
    "huggingface-hub==0.27.1", "accelerate==1.2.1", "diffusers==0.32.2",
    "addict==2.4.0", "yapf==0.40.2", "pycocotools==2.0.7",
    "einops==0.8.0", "einshape==1.0", "mediapy==1.2.0", "av==12.0.0",
)
CUDA_PIP_OPTIONS = ("--extra-index-url", "https://download.pytorch.org/whl/cu121")


def git_blob_hash(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()  # noqa: S324


def verify_source(root: Path, blobs: Optional[Mapping[str, str]] = None) -> None:
    blobs = PINNED_BLOBS if blobs is None else blobs
    for relative, expected in blobs.items():
        path = root / relative
        if not path.is_file() or git_blob_hash(path) != expected:
            raise ValueError(f"T2V-CompBench source does not match {REVISION}: {path}")


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination_resolved = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            roots = {Path(item.filename).parts[0] for item in members if Path(item.filename).parts}
            if len(roots) != 1:
                raise ValueError("T2V-CompBench archive must contain one root directory")
            for item in members:
                target = (destination / item.filename).resolve()
                if target != destination_resolved and destination_resolved not in target.parents:
                    raise ValueError(f"unsafe T2V-CompBench archive member: {item.filename}")
                if item.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError(f"symlink is not allowed in source archive: {item.filename}")
            bundle.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid T2V-CompBench source archive: {archive}") from exc
    return destination / roots.pop()


def ensure_source(
    models_dir: Path,
    *,
    downloader: Callable[[str, Path], object] = urllib.request.urlretrieve,
) -> Path:
    cache = Path(models_dir).expanduser().resolve() / ".ayase" / "sources"
    target = cache / f"t2v-compbench-{REVISION}"
    cache.mkdir(parents=True, exist_ok=True)
    with _SetupLock(cache / f"t2v-compbench-{REVISION}.lock"):
        if target.is_dir():
            try:
                verify_source(target)
                return target
            except ValueError:
                shutil.rmtree(target)
        with tempfile.TemporaryDirectory(prefix="t2v-source-", dir=cache) as temp_name:
            temp = Path(temp_name)
            archive = temp / "source.zip"
            downloader(SOURCE_URL, archive)
            extracted = _safe_extract(archive, temp / "unpacked")
            verify_source(extracted)
            _enable_groundingdino_fallback(extracted)
            _trim_llava_training_dependencies(extracted)
            extracted.replace(target)
    return target


def _enable_groundingdino_fallback(root: Path) -> None:
    """Use upstream's exact PyTorch operator when the optional extension is absent."""

    path = root / (
        "Grounded-Segment-Anything/GroundingDINO/groundingdino/models/"
        "GroundingDINO/ms_deform_attn.py"
    )
    text = path.read_text(encoding="utf-8")
    original = "try:\n    from groundingdino import _C\nexcept:\n"
    replacement = "try:\n    from groundingdino import _C\nexcept Exception:\n    _C = None\n"
    if original in text:
        text = text.replace(original, replacement, 1)
    condition = "if value.is_cuda:\n"
    if condition in text:
        text = text.replace(condition, "if value.is_cuda and _C is not None:\n", 1)
    path.write_text(text, encoding="utf-8")


def _trim_llava_training_dependencies(root: Path) -> None:
    """Remove optional compiled/training packages unused by CompBench inference."""

    path = root / "LLaVA" / "pyproject.toml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for fragment in (
        '"bitsandbytes", ',
        '"cmake==3.29.2", ',
        '"sglang==0.1.14", ',
        '"triton==2.3.0", ',
        '"xformers==0.0.26.post1", ',
        '"peft", ',
    ):
        text = text.replace(fragment, "")
    path.write_text(text, encoding="utf-8")


def _download_asset(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".part", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size == 0:
            raise ValueError(f"empty checkpoint download: {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_assets(
    root: Path,
    categories: Iterable[str],
    *,
    downloader: Callable[[str, Path], object] = _download_asset,
) -> None:
    selected = set(categories)
    groups = set()
    if selected & {"spatial_relationships", "generative_numeracy", "motion_binding"}:
        groups.add("groundingdino")
    if selected & {"spatial_relationships", "motion_binding"}:
        groups.add("sam")
    if "motion_binding" in selected:
        groups.add("dot")
    for relative, (url, group) in ASSETS.items():
        target = root / relative
        if group in groups:
            with _SetupLock(target.with_suffix(target.suffix + ".lock")):
                if not target.is_file() or target.stat().st_size == 0:
                    downloader(url, target)


def ensure_environments(
    root: Path,
    categories: Iterable[str],
    models_dir: Path,
    *,
    provisioner: Callable[..., Path] = ensure_managed_environment,
    python_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    selected = set(categories)
    overrides = dict(python_overrides or {})
    result: Dict[str, Path] = {}
    llava_categories = {
        "consistent_attribute_binding", "dynamic_attribute_binding",
        "action_binding", "object_interactions",
    }
    if selected & llava_categories:
        result["llava"] = provisioner(
            "t2v-compbench-llava", [*LLAVA_REQUIREMENTS, root / "LLaVA"], models_dir,
            python=overrides.get("llava"),
            python_version=None if overrides.get("llava") else "3.10.15",
            pip_options=CUDA_PIP_OPTIONS,
            source_roots=[root / "LLaVA"],
        )
    if selected - llava_categories:
        requirements: Sequence[object] = [
            *DETECTION_REQUIREMENTS,
            root / "Grounded-Segment-Anything" / "segment_anything",
        ]
        result["vision"] = provisioner(
            "t2v-compbench-vision", requirements, models_dir,
            python=overrides.get("vision"),
            python_version=None if overrides.get("vision") else "3.12.3",
            pip_options=CUDA_PIP_OPTIONS,
            source_roots=[root / "Grounded-Segment-Anything", root / "Depth-Anything", root / "dot"],
        )
    return result


def gpu_is_available() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        probe = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())
