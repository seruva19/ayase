"""Pinned isolated-runtime dependencies for published metric backends.

This registry covers modules whose primary implementation imports a package that
is intentionally absent from Ayase's base environment. It is consumed by the
managed worker bridge; entries are not instructions to modify the host Python.
Bundled implementations need no entry; optional backends are declared with configuration conditions.
"""

from dataclasses import dataclass
import re
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class BackendDependency:
    """Reproducible dependency set for one module's primary backend."""

    module: str
    requirements: Tuple[str, ...]
    python_version: str = "3.11"
    pip_options: Tuple[str, ...] = ()
    no_deps_requirements: Tuple[str, ...] = ()
    conda_packages: Tuple[str, ...] = ()
    source_url: Optional[str] = None
    source_revision: Optional[str] = None
    config_conditions: Tuple[Tuple[str, str], ...] = ()
    unsupported_reason: Optional[str] = None
    unsupported_platforms: Tuple[Tuple[str, str], ...] = ()
    cpu_only: bool = False


COMMON_MANAGED_REQUIREMENTS = (
    "numpy==1.26.4",
    "scipy==1.11.4",
    "torch==2.6.0+cu124",
    "torchvision==0.21.0+cu124",
    "torchaudio==2.6.0+cu124",
    "transformers==4.57.6",
    "huggingface-hub==0.36.0",
    "pillow==10.4.0",
    "opencv-python==4.10.0.84",
    "librosa==0.10.2.post1",
    "soundfile==0.12.1",
)
COMMON_PIP_OPTIONS = ("--extra-index-url", "https://download.pytorch.org/whl/cu124")


def _pypi(
    module: str, *requirements: str, python_version: str = "3.11", cpu_only: bool = False
) -> BackendDependency:
    return BackendDependency(module, requirements, python_version=python_version, cpu_only=cpu_only)


def _archive(module: str, package: str, repository: str, revision: str) -> BackendDependency:
    archive = f"{repository}/archive/{revision}.zip"
    requirement = f"{package} @ {archive}"
    return BackendDependency(
        module,
        (requirement,),
        source_url=repository,
        source_revision=revision,
    )


_ENTRIES = (
    _pypi("pqsg", "google-genai==1.31.0", "openai==2.7.2", cpu_only=True),
    BackendDependency(
        "avc_reward",
        (
            "qwen-omni-utils==0.0.8",
            "av==15.1.0",
            "accelerate==1.12.0",
            "tqdm==4.66.5",
            "ftfy==6.3.1",
        ),
        conda_packages=("ffmpeg=7.1.1",),
    ),
    _pypi(
        "palate",
        "omegaconf==2.3.0",
        "ftfy==6.3.1",
    ),
    _pypi(
        "chd",
        "einops==0.8.1",
        "omegaconf==2.3.0",
        "safetensors==0.4.5",
        "tqdm==4.66.5",
        "ftfy==6.3.1",
    ),
    _pypi(
        "beyond_time_shifts",
        "transformers==4.57.1",
        "qwen-omni-utils==0.0.8",
        "av==15.1.0",
        "accelerate==1.12.0",
        "tqdm==4.66.5",
        "ftfy==6.3.1",
    ),
    _pypi(
        "anyaudio_judge",
        "accelerate==1.12.0",
        "sentencepiece==0.2.0",
        "ftfy==6.3.1",
    ),
    _pypi("active_speaker", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("adaface", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("aesthetic", "aesthetic-predictor-v2-5==2024.12.18.1"),
    _pypi("artfid", "art-fid==0.0.2"),
    _pypi("aucon_prmse", "py-feat==2.1.5"),
    _pypi("audio_estoi", "pystoi==0.4.1", cpu_only=True),
    _pypi("audio_isc", "torchlibrosa==0.1.0", "hear21passt==0.0.26"),
    _pypi("audio_kl", "torchlibrosa==0.1.0", "hear21passt==0.0.26"),
    BackendDependency(
        "audio_log_f0_dtw",
        (),
        conda_packages=("pyworld=0.3.5", "fastdtw=0.3.4"),
        cpu_only=True,
    ),
    BackendDependency(
        "audio_mcd",
        (),
        conda_packages=("pyworld=0.3.5", "fastdtw=0.3.4"),
        cpu_only=True,
    ),
    BackendDependency("audio_pesq", (), conda_packages=("pesq=0.0.4",), cpu_only=True),
    _pypi(
        "av_desync",
        "insightface==2.1",
        "onnxruntime==1.31.0",
        "moviepy==1.0.3",
    ),
    _archive(
        "audio_utmos_v2",
        "utmosv2",
        "https://github.com/sarulab-speech/UTMOSv2",
        "cc2700db57bb83ee13dc31ebe1b868c254e15d09",
    ),
    _pypi("background_diversity", "rembg==2.0.86", "onnxruntime==1.31.0", cpu_only=True),
    _pypi("captioning", "pycocoevalcap==1.2"),
    BackendDependency(
        "camera_trajectory",
        (
            "torch==2.3.1+cu121",
            "torchvision==0.18.1+cu121",
            "torchaudio==2.3.1+cu121",
            "numpy==1.26.1",
            "vggt @ https://github.com/facebookresearch/vggt/archive/"
            "a288dd0f14786c93483e45524328726ab7b1b4ce.zip",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu121"),
        source_url="https://github.com/facebookresearch/vggt",
        source_revision="a288dd0f14786c93483e45524328726ab7b1b4ce",
        config_conditions=(("pose_backend", "vggt"),),
    ),
    _pypi("celebrity_id", "deepface==0.0.103"),
    _pypi("concept_presence", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("cpbd", "cpbd==1.0.7"),
    _pypi("csim", "insightface==2.1", "onnxruntime==1.31.0"),
    BackendDependency(
        "cycle_reward",
        (
            "transformers==4.37.2",
            "timm==0.9.16",
            "fairscale==0.4.13",
        ),
        # CycleReward 0.1.7 declares Triton but never imports it.  Triton has no
        # Windows wheels, so install the exact upstream wheel after its complete
        # inference stack without resolving that erroneous metadata dependency.
        no_deps_requirements=(
            "cyclereward @ https://files.pythonhosted.org/packages/5c/5c/"
            "950d74cb2a8b3145893fd7042c13718709355934d6b70222bfff8d323611/"
            "cyclereward-0.1.7-py3-none-any.whl#sha256="
            "4a361e21454c7871f2393798e868af028888443a76ae5a983938d53ee273d095",
        ),
    ),
    _pypi("dists", "piq==0.8.0"),
    BackendDependency(
        "deqa_score",
        (
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2+cu118",
            "transformers==4.36.1",
            "tokenizers==0.15.0",
            "sentencepiece==0.1.99",
            "accelerate==0.21.0",
            "icecream==2.1.3",
            "protobuf==3.20.3",
            "safetensors==0.4.5",
            "ftfy==6.3.1",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu118"),
    ),
    _pypi("dedup", "ImageHash==4.3.2", cpu_only=True),
    _pypi("dino_face_identity", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("dreamsim", "dreamsim==0.2.1"),
    _pypi(
        "edit_score",
        "accelerate==1.12.0",
        "peft==0.17.1",
        "json-repair==0.30.3",
        "regex==2024.11.6",
        "ftfy==6.3.1",
    ),
    _pypi("entity_consistency", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("erqa", "erqa==1.1.2"),
    _pypi(
        "fad",
        "fadtk==1.1.0",
        "panns-inference==0.1.1",
        "hear21passt==0.0.26",
    ),
    _pypi("flip", "flip-evaluator==1.7"),
    _pypi("flolpips", "ptlflow==0.4.2"),
    _pypi(
        "face_cross_similarity",
        "insightface==2.1",
        "onnxruntime==1.31.0",
        "deepface==0.0.103",
    ),
    _pypi("face_emb_norm", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("face_identity_drift", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("facesim", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("grafiqs", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi(
        "geneval2",
        "transformers==4.57.0",
        "accelerate==1.12.0",
        "ftfy==6.3.1",
    ),
    _pypi("hpsv2", "hpsv2==1.2.0"),
    _pypi("hpsv2_const", "hpsv2==1.2.0"),
    BackendDependency(
        "image_reward",
        (
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2+cu118",
            "transformers==4.37.2",
            "image-reward==1.5",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu118"),
    ),
    _pypi("inception_score", "torch-fidelity==0.4.0"),
    _pypi("kad", "kadtk==1.1.0"),
    BackendDependency(
        "morpheus",
        (
            "pandas==2.2.3",
            "matplotlib==3.9.4",
            "scikit-learn==1.6.1",
            "scikit-image==0.24.0",
            "natsort==8.4.0",
            "imageio==2.37.0",
            "imageio-ffmpeg==0.6.0",
            "hydra-core==1.3.2",
            "iopath==0.1.10",
            "tqdm==4.67.1",
            "ftfy==6.3.1",
        ),
    ),
    _pypi("kid", "clean-fid==0.1.35", "torch-fidelity==0.4.0"),
    _pypi("identity_loss", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi(
        "mauve_audio_divergence",
        "mauve-text==0.4.0",
        "faiss-cpu==1.15.1",
    ),
    _pypi("ms_ssim", "pytorch-msssim==1.0.0"),
    _pypi("multi_subject_identity", "insightface==2.1", "onnxruntime==1.31.0"),
    _pypi("chamfer_sim", "open3d==0.20.0", cpu_only=True),
    _pypi("local_std_uniformity", "open3d==0.20.0", cpu_only=True),
    _pypi("mse_color", "open3d==0.20.0", cpu_only=True),
    _pypi("pc_psnr", "open3d==0.20.0", cpu_only=True),
    BackendDependency(
        "object_integrity",
        (
            "torch==2.1.2+cu121",
            "torchvision==0.16.2+cu121",
            "torchaudio==2.1.2+cu121",
            "transformers==4.37.2",
            "chumpy @ https://github.com/mattloper/chumpy/archive/"
            "580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip",
            "mmcv==2.1.0",
            "mmdet==3.3.0",
            "mmpose==1.3.2",
        ),
        pip_options=(
            "--find-links",
            "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu121",
            "--no-build-isolation",
        ),
        config_conditions=(("backend", "mmpose"),),
    ),
    _archive(
        "p1203",
        "itu-p1203",
        "https://github.com/itu-p1203/itu-p1203",
        "124b07dc4ffdcb11ea45f995476ddcf92e4dc28a",
    ),
    _pypi("perceptual_fr", "piq==0.8.0"),
    BackendDependency(
        "opens2v",
        (
            "mmyolo==0.6.0",
            "yolo-world @ https://github.com/AILab-CVC/YOLO-World/archive/"
            "4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b.zip",
        ),
        source_url="https://github.com/AILab-CVC/YOLO-World",
        source_revision="4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b",
    ),
    _pypi("ptlflow_motion", "ptlflow==0.4.2"),
    _archive(
        "ram_tagging",
        "recognize-anything",
        "https://github.com/xinyu1205/recognize-anything",
        "7cb804a8609e9f4b1a50b7f31436d2df40bb9481",
    ),
    _pypi("scene_detection", "transnetv2-pytorch==1.0.5"),
    _pypi("scoreq", "scoreq==1.0.1"),
    BackendDependency(
        "serfiq",
        (
            "numpy==1.23.5",
            "scipy==1.11.4",
            "scikit-image==0.22.0",
            "insightface==2.1",
            "onnxruntime==1.31.0",
        ),
        no_deps_requirements=("mxnet==1.7.0.post2",),
        cpu_only=True,
    ),
    _pypi("ssimulacra2", "ssimulacra2==0.3.0"),
    _pypi("stlpips_selfdist", "stlpips-pytorch==0.0.2"),
    _pypi("tifa", "word2number==1.1", "modelscope==1.40.2"),
    _pypi("tts_system_dist", "ttsds==2.1.3"),
    _pypi("vendi", "vendi-score==0.0.3"),
    BackendDependency(
        "voice_identity",
        (
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "torchaudio==2.7.1+cu128",
            "speechbrain==1.1.2",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu128"),
    ),
    BackendDependency(
        "voice_identity_drift",
        (
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "torchaudio==2.7.1+cu128",
            "speechbrain==1.1.2",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu128"),
    ),
    _pypi(
        "worldmodelbench",
        "hydra-core==1.3.7",
        "timm==1.0.22",
        "einops==0.8.1",
    ),
    BackendDependency(
        "videophy2_autoeval",
        (
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2+cu118",
            "transformers==4.28.1",
            "einops==0.6.1",
            "decord==0.6.0",
            "sentencepiece==0.1.99",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu118"),
    ),
    _pypi(
        "visqol",
        "visqol-python==3.8.0",
        "libsvm-official==3.37.0",
        "ai-edge-litert==2.3.0",
        cpu_only=True,
    ),
    BackendDependency(
        "syncreward",
        (
            "torch==2.9.1+cu128",
            "torchvision==0.24.1+cu128",
            "torchaudio==2.9.1+cu128",
            "torchcodec==0.8.1",
            "transformers==5.1.0",
            "timm==1.0.26",
            "safetensors==0.8.0",
            "huggingface-hub==1.33.0",
            "PyYAML==6.0.2",
            "ftfy==6.3.1",
        ),
        pip_options=("--extra-index-url", "https://download.pytorch.org/whl/cu128"),
        conda_packages=("ffmpeg=7.1.1",),
    ),
    _pypi("vif", "piq==0.8.0"),
    _pypi(
        "verse_bench",
        "hear21passt==0.0.26",
        "audiobox-aesthetics==0.0.4",
        "msclap==1.3.4",
    ),
)


BACKEND_DEPENDENCIES: Dict[str, BackendDependency] = {
    dependency.module: dependency for dependency in _ENTRIES
}


def get_backend_dependency(module_name: str) -> Optional[BackendDependency]:
    """Return isolated-runtime requirements for ``module_name``, when declared."""

    return BACKEND_DEPENDENCIES.get(module_name)


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"invalid requirement: {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def managed_requirements(dependency: BackendDependency) -> Tuple[str, ...]:
    """Return a self-contained inference stack, honoring backend-specific pins."""

    overrides = {_requirement_name(item) for item in dependency.requirements}
    excluded = {"torch", "torchvision", "torchaudio", "transformers", "huggingface-hub"}
    common = tuple(
        item
        for item in COMMON_MANAGED_REQUIREMENTS
        if _requirement_name(item) not in overrides
        and not (getattr(dependency, "cpu_only", False) and _requirement_name(item) in excluded)
    )
    return common + dependency.requirements


def managed_pip_options(dependency: BackendDependency) -> Tuple[str, ...]:
    """Return the CUDA wheel index plus any backend-specific package links."""

    if "--extra-index-url" in dependency.pip_options:
        return dependency.pip_options
    if getattr(dependency, "cpu_only", False):
        return dependency.pip_options
    return COMMON_PIP_OPTIONS + dependency.pip_options


__all__ = [
    "BACKEND_DEPENDENCIES",
    "COMMON_MANAGED_REQUIREMENTS",
    "COMMON_PIP_OPTIONS",
    "BackendDependency",
    "get_backend_dependency",
    "managed_pip_options",
    "managed_requirements",
]
