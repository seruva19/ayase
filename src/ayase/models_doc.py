"""Generate MODELS.md — catalog of all ML models, weights, and external assets.

Extracts model references from module source code:
- HuggingFace model IDs (from_pretrained patterns)
- pyiqa metric names (pyiqa.create_metric)
- torch.hub repos (torch.hub.load)
- torchvision pretrained weights
- CLIP/OpenCLIP model variants
- FFmpeg/libvmaf models
- Local/downloaded weight files
"""

import inspect
import json
import logging
import re
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .pipeline import ModuleRegistry


def _validate_urls(entries: Dict[str, "ModelEntry"]) -> List[str]:
    """Check that model URLs are reachable. Returns list of warnings."""
    import sys
    warnings = []
    urls_to_check = []
    for key, entry in entries.items():
        if entry.url and entry.url.startswith("http"):
            urls_to_check.append((key, entry.url))

    if not urls_to_check:
        return warnings

    print(f"Validating {len(urls_to_check)} model URLs...", file=sys.stderr)
    for key, url in urls_to_check:
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "ayase-models-doc/1.0")
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status >= 400:
                warnings.append(f"  {key}: HTTP {resp.status} — {url}")
        except urllib.error.HTTPError as e:
            warnings.append(f"  {key}: HTTP {e.code} — {url}")
        except Exception as e:
            warnings.append(f"  {key}: {type(e).__name__} — {url}")
    return warnings

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A single model/weight file used by one or more modules."""
    name: str
    source: str  # "huggingface", "pyiqa", "torch_hub", "torchvision", "clip", "ffmpeg", "local", "pip"
    modules: List[str] = field(default_factory=list)
    url: Optional[str] = None
    install: Optional[str] = None  # pip install command
    size_estimate: Optional[str] = None
    vram_estimate: Optional[str] = None
    task: Optional[str] = None  # what it does
    auto_download: bool = True
    notes: Optional[str] = None
    license: Optional[str] = None
    commercial_ok: Optional[bool] = None  # True=OK, False=restricted, None=unknown
    # HF-fetched metadata
    downloads: Optional[int] = None
    likes: Optional[int] = None
    parameters: Optional[int] = None  # from safetensors
    pipeline_tag: Optional[str] = None
    library: Optional[str] = None
    arxiv: Optional[str] = None


# ── Size estimates for known models ──────────────────────────────────────────
_SIZE_DB = {
    # HuggingFace models
    "openai/clip-vit-base-patch32": ("~600 MB", "~600 MB"),
    "openai/clip-vit-large-patch14": ("~1.7 GB", "~1.5 GB"),
    "laion/clap-htsat-fused": ("~600 MB", "~600 MB"),
    "MCG-NJU/videomae-large-finetuned-kinetics": ("~1.3 GB", "~1.5 GB"),
    "MCG-NJU/videomae-base-finetuned-kinetics": ("~350 MB", "~400 MB"),
    "dandelin/vilt-b32-finetuned-vqa": ("~450 MB", "~500 MB"),
    "llava-hf/llava-1.5-7b-hf": ("~14 GB", "~14 GB"),
    "llava-hf/llava-v1.6-mistral-7b-hf": ("~14 GB", "~14 GB"),
    "microsoft/xclip-base-patch32": ("~600 MB", "~600 MB"),
    "Salesforce/blip-image-captioning-base": ("~990 MB", "~1 GB"),
    "Salesforce/blip2-opt-2.7b": ("~6 GB", "~6 GB"),
    "depth-anything/Depth-Anything-V2-Small-hf": ("~100 MB", "~200 MB"),
    "intel-isl/MiDaS": ("~400 MB", "~400 MB"),
    "TIGER-Lab/VideoScore": ("~14 GB", "~14 GB"),
    "google/vit-base-patch16-224": ("~350 MB", "~400 MB"),
    "facebook/dinov2-vitb14": ("~350 MB", "~400 MB"),
    # pyiqa models (downloaded on first use)
    "pyiqa:brisque": ("~1 MB", "~50 MB"),
    "pyiqa:niqe": ("~1 MB", "~50 MB"),
    "pyiqa:clipiqa+": ("~600 MB", "~600 MB"),
    "pyiqa:maniqa": ("~150 MB", "~300 MB"),
    "pyiqa:topiq_nr": ("~150 MB", "~300 MB"),
    "pyiqa:topiq_fr": ("~150 MB", "~300 MB"),
    "pyiqa:dbcnn": ("~100 MB", "~200 MB"),
    "pyiqa:musiq": ("~150 MB", "~300 MB"),
    "pyiqa:nima": ("~100 MB", "~200 MB"),
    "pyiqa:hyperiqa": ("~100 MB", "~200 MB"),
    "pyiqa:ahiq": ("~150 MB", "~300 MB"),
    # torch.hub
    "tarepan/SpeechMOS:v1.2.0": ("~100 MB", "~200 MB"),
    # torchvision
    "torchvision:raft_small": ("~20 MB", "~100 MB"),
    "torchvision:raft_large": ("~20 MB", "~200 MB"),
    "torchvision:r3d_18": ("~130 MB", "~200 MB"),
    "torchvision:resnet18": ("~45 MB", "~100 MB"),
    "torchvision:inception_v3": ("~100 MB", "~200 MB"),
    # CLIP
    "clip:ViT-B/32": ("~340 MB", "~600 MB"),
    "clip:ViT-L/14": ("~900 MB", "~1.5 GB"),
    # FFmpeg
    "ffmpeg:vmaf_v0.6.1": ("built-in", "N/A"),
    "ffmpeg:vmaf_4k_v0.6.1": ("built-in", "N/A"),
    "ffmpeg:xpsnr": ("built-in", "N/A"),
    "ffmpeg:cambi": ("built-in", "N/A"),
}

_TASK_DB = {
    "brisque": "Blind naturalness statistics NR-IQA",
    "niqe": "Natural image quality evaluator (statistics-based)",
    "ilniqe": "Integrated local NIQE",
    "piqe": "Perception-based blind NR-IQA",
    "nrqm": "No-reference quality metric",
    "pi": "Perceptual index (PIRM challenge)",
    "clipiqa+": "CLIP-based image quality assessment",
    "clip_iqa": "CLIP image quality assessment",
    "maniqa": "Multi-dimension attention NR-IQA",
    "topiq_nr": "Transformer-based NR image quality",
    "topiq_fr": "Transformer-based FR image quality",
    "topiq": "TOPIQ transformer quality",
    "topiq_nr-face": "TOPIQ face-specific quality",
    "dbcnn": "Deep bilinear CNN for blind IQA",
    "musiq": "Multi-scale image quality transformer",
    "nima": "Neural image assessment (aesthetic + technical)",
    "hyperiqa": "Adaptive hypernetwork NR-IQA",
    "ahiq": "Attention-based hybrid FR-IQA",
    "contrique": "Contrastive image quality representation",
    "cnniqa": "CNN-based blind image quality",
    "arniqa": "Artifact-aware NR-IQA",
    "qualiclip": "Quality-aware CLIP embeddings",
    "liqe": "Learned image quality evaluator (multi-task)",
    "dover": "Disentangled objective video evaluation",
    "cover": "Comprehensive video evaluation and rating",
    "finevq": "Fine-grained UGC video quality",
    "kvq": "Key-frame saliency-guided VQA",
    "rqvqa": "Rich quality-aware VQA",
    "compare2score": "Comparative-to-absolute quality scoring",
    "unique": "Unified NR-IQA with contrastive learning",
    "tres": "Transformer relative quality estimation",
    "laion_aes": "LAION aesthetic scoring (CLIP-based)",
    "laion_aesthetic": "LAION Aesthetics V2 predictor",
    "paq2piq": "Patches-as-questions for image quality",
    "maclip": "Multi-attribute CLIP quality scoring",
    "mdtvsfa": "Multi-dimensional temporal-spatial VQA",
    "deepwsd": "Deep Wasserstein distance IQA",
    "wadiqam_nr": "Weighted average deep NR-IQA",
    "wadiqam_fr": "Weighted average deep FR-IQA",
    "wadiqam": "Weighted average deep IQA",
    "ckdn": "Conditional knowledge distillation FR-IQA",
    "dmm": "Detail model metric FR-IQA",
    "ssimc": "Complex wavelet SSIM-C FR",
    "cw_ssim": "Complex wavelet SSIM",
    "nlpd": "Normalized Laplacian pyramid distance",
    "pieapp": "Pairwise learned perceptual distance",
    "mad": "Most apparent distortion FR-IQA",
    "deepwsd": "Deep Wasserstein distance FR-IQA",
    "promptiqa": "Few-shot prompt-based NR-IQA",
    "qcn": "Geometric order blind IQA",
    "deepdc": "Deep distribution conformance",
    "sfid": "Spatial FID distribution metric",
    "msswd": "Multi-scale sliced Wasserstein distance",
    "afine": "Adaptive fidelity-naturalness IQA",
    "afine_nr": "A-FINE NR fidelity-naturalness",
    "bvqi": "Zero-shot blind VQA",
    "conviqt": "Contrastive NR-VQA",
    "creativity": "Creative quality assessment",
    "face_iqa": "TOPIQ face-specific quality",
    "naturalness": "Natural scene statistics",
}


# ── Known licenses (hardcoded for non-HF sources) ───────────────────────────
_KNOWN_LICENSES: Dict[str, tuple] = {
    # (license_name, commercial_ok)
    # pyiqa — MIT-licensed library, but underlying models vary
    # Most pyiqa models are research code with permissive or unspecified licenses
    "pyiqa:brisque": ("BSD-2-Clause (OpenCV)", True),
    "pyiqa:niqe": ("BSD-2-Clause (OpenCV)", True),
    "pyiqa:ilniqe": ("BSD-2-Clause", True),
    "pyiqa:piqe": ("BSD-2-Clause", True),
    "pyiqa:nrqm": ("research", None),
    "pyiqa:pi": ("research", None),
    "pyiqa:clipiqa+": ("MIT (pyiqa)", True),
    "pyiqa:maniqa": ("Apache-2.0", True),
    "pyiqa:topiq_nr": ("MIT (pyiqa)", True),
    "pyiqa:topiq_fr": ("MIT (pyiqa)", True),
    "pyiqa:dbcnn": ("research", None),
    "pyiqa:musiq": ("Apache-2.0 (Google)", True),
    "pyiqa:nima": ("Apache-2.0 (Google)", True),
    "pyiqa:hyperiqa": ("research", None),
    "pyiqa:ahiq": ("research", None),
    "pyiqa:contrique": ("research", None),
    "pyiqa:cnniqa": ("research", None),
    "pyiqa:paq2piq": ("research", None),
    "pyiqa:tres": ("research", None),
    "pyiqa:laion_aes": ("MIT", True),
    "pyiqa:pieapp": ("research", None),
    "pyiqa:mad": ("research", None),
    "pyiqa:nlpd": ("research", None),
    "pyiqa:cw_ssim": ("MIT (pyiqa)", True),
    "pyiqa:ssimc": ("MIT (pyiqa)", True),
    "pyiqa:deepwsd": ("research", None),
    "pyiqa:ckdn": ("research", None),
    "pyiqa:dmm": ("research", None),
    "pyiqa:wadiqam_nr": ("research", None),
    "pyiqa:wadiqam_fr": ("research", None),
    "pyiqa:dover": ("MIT (pyiqa)", True),
    "pyiqa:unique": ("research", None),
    "pyiqa:compare2score": ("research", None),
    "pyiqa:afine_nr": ("research", None),
    "pyiqa:maclip": ("research", None),
    "pyiqa:qualiclip": ("research", None),
    "pyiqa:arniqa": ("research", None),
    "pyiqa:promptiqa": ("research", None),
    "pyiqa:qcn": ("research", None),
    "pyiqa:topiq_nr-face": ("MIT (pyiqa)", True),
    # torchvision — all BSD
    "tv:raft_small": ("BSD-3-Clause", True),
    "tv:raft_large": ("BSD-3-Clause", True),
    "tv:r3d_18": ("BSD-3-Clause", True),
    "tv:resnet18": ("BSD-3-Clause", True),
    "tv:inception_v3": ("BSD-3-Clause", True),
    "tv:video": ("BSD-3-Clause", True),
    # CLIP
    "clip:ViT-B/32": ("MIT (OpenAI)", True),
    "clip:ViT-L/14": ("MIT (OpenAI)", True),
    # torch.hub
    "hub:facebookresearch/co-tracker": ("Apache-2.0", True),
    "hub:facebookresearch/dinov2": ("Apache-2.0", True),
    "hub:intel-isl/MiDaS": ("MIT", True),
    "hub:tarepan/SpeechMOS:v1.2.0": ("MIT", True),
    # FFmpeg
    "ff:libvmaf": ("BSD-2-Clause (Netflix)", True),
    "ff:vmaf_v0.6.1": ("BSD-2-Clause (Netflix)", True),
    "ff:vmaf_4k_v0.6.1": ("BSD-2-Clause (Netflix)", True),
    "ff:vmaf_v0.6.1neg": ("BSD-2-Clause (Netflix)", True),
    "ff:vmaf_phone_model": ("BSD-2-Clause (Netflix)", True),
    "ff:xpsnr": ("BSD (FFmpeg)", True),
    "ff:cambi": ("BSD-2-Clause (Netflix)", True),
}

# Licenses known to be commercial-friendly
_COMMERCIAL_OK_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd",
    "isc", "unlicense", "cc0-1.0", "wtfpl", "0bsd", "artistic-2.0",
    "zlib", "bsl-1.0", "ecl-2.0",
}

# Licenses known to restrict commercial use
_NON_COMMERCIAL_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
    "cc-by-nc-2.0", "cc-by-nc-sa-2.0", "cc-by-nc-3.0",
    "non-commercial", "research-only", "academic",
}


def _classify_license(license_str: Optional[str]) -> Optional[bool]:
    """Classify a license string as commercial-OK (True), restricted (False), or unknown (None)."""
    if not license_str:
        return None
    low = license_str.lower().replace(" ", "-")
    if any(ok in low for ok in _COMMERCIAL_OK_LICENSES):
        return True
    if any(nc in low for nc in _NON_COMMERCIAL_LICENSES):
        return False
    return None


def _fetch_hf_license(model_id: str) -> Optional[str]:
    """Query HuggingFace API for model license. Returns license string or None."""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ayase-models-doc/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            # License can be in cardData.license, tags, or top-level
            if "cardData" in data and isinstance(data["cardData"], dict):
                lic = data["cardData"].get("license")
                if lic:
                    return str(lic)
            # Check tags for license
            for tag in data.get("tags", []):
                if tag.startswith("license:"):
                    return tag.split(":", 1)[1]
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        logger.debug("HF API query failed for %s: %s", model_id, e)
        return None


def _fetch_hf_licenses_batch(model_ids: List[str]) -> Dict[str, str]:
    """Fetch licenses for multiple HF models. Returns {model_id: license}."""
    result = {}
    for model_id in model_ids:
        if "/" in model_id and model_id.count("/") == 1:
            lic = _fetch_hf_license(model_id)
            if lic:
                result[model_id] = lic
    return result


def _fetch_hf_info_batch(model_ids: List[str]) -> Dict[str, dict]:
    """Fetch rich metadata for HF models via huggingface_hub API.

    Returns {model_id: {license, downloads, likes, parameters, pipeline_tag, library, arxiv}}.
    """
    result: Dict[str, dict] = {}
    try:
        from huggingface_hub import model_info as _model_info
    except ImportError:
        logger.debug("huggingface_hub not installed, skipping HF metadata fetch")
        return result

    for model_id in model_ids:
        if "/" not in model_id or model_id.count("/") != 1:
            continue
        try:
            info = _model_info(model_id)
            data: dict = {}
            # License
            lic = None
            if info.card_data and info.card_data.license:
                lic = info.card_data.license
            elif info.tags:
                for tag in info.tags:
                    if tag.startswith("license:"):
                        lic = tag.split(":", 1)[1]
                        break
            data["license"] = lic
            # Downloads & likes
            data["downloads"] = info.downloads
            data["likes"] = info.likes
            # Parameters from safetensors
            params = None
            if info.safetensors and hasattr(info.safetensors, "total"):
                params = info.safetensors.total
            elif info.safetensors and hasattr(info.safetensors, "parameters"):
                p = info.safetensors.parameters
                if isinstance(p, dict):
                    params = sum(p.values())
                elif isinstance(p, int):
                    params = p
            data["parameters"] = params
            # Pipeline tag and library
            data["pipeline_tag"] = info.pipeline_tag
            data["library"] = info.library_name
            # ArXiv from tags
            arxiv = None
            if info.tags:
                for tag in info.tags:
                    if tag.startswith("arxiv:"):
                        arxiv = tag.split(":", 1)[1]
                        break
            data["arxiv"] = arxiv
            result[model_id] = data
            logger.info("  %s → %s, %s downloads, %s params",
                        model_id, lic, data["downloads"], data["parameters"])
        except Exception as exc:
            logger.debug("Failed to fetch HF info for %s: %s", model_id, exc)
    return result


def _get_source(cls) -> str:
    try:
        return inspect.getsource(cls)
    except (TypeError, OSError):
        return ""


def _get_module_source(cls) -> str:
    """Get full module-level source (includes module-level constants/dicts)."""
    try:
        mod = inspect.getmodule(cls)
        if mod is not None:
            return inspect.getsource(mod)
    except (TypeError, OSError):
        pass
    return _get_source(cls)


def _extract_hf_models(source: str, default_config: dict = None) -> List[str]:
    """Extract HuggingFace model IDs from from_pretrained() calls and config defaults."""
    models = set()
    # Direct from_pretrained calls with string literals
    for m in re.finditer(r'from_pretrained\s*\(\s*["\']([a-zA-Z0-9_/-]+)["\']', source):
        candidate = m.group(1)
        if "/" in candidate and not any(x in candidate for x in ("http", "path", ".py")):
            models.add(candidate)
    # Variable-based: model_name = "org/model" followed by from_pretrained(model_name)
    for m in re.finditer(r'model_name\s*=\s*["\']([a-zA-Z0-9_/-]+)["\']', source):
        candidate = m.group(1)
        if "/" in candidate and "from_pretrained" in source:
            models.add(candidate)
    # Quoted org/model strings near from_pretrained (catches indirect references)
    for m in re.finditer(r'["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']', source):
        candidate = m.group(1)
        if ("from_pretrained" in source or "AutoModel" in source) and \
           not any(x in candidate for x in ("http", "path", ".py", ".pth", ".onnx",
                                            "ayase-models", "resolve/main",
                                            "models/", "subfolder",
                                            "facebookresearch/", "intel-isl/",
                                            "tarepan/")):
            models.add(candidate)
    # Model names from default_config
    if default_config:
        for key in ("model_name", "vlm_model", "vlm_model_name", "clip_model",
                     "sdxl_model", "vqa_model", "xclip_model_name"):
            val = default_config.get(key, "")
            if isinstance(val, str) and "/" in val and not val.startswith(("http", "/")):
                models.add(val)
    return sorted(models)


def _extract_pyiqa_metrics(source: str) -> List[str]:
    """Extract pyiqa.create_metric() metric names."""
    metrics = set()
    # Direct string: create_metric("name")
    for m in re.finditer(r'create_metric\s*\(\s*["\']([a-zA-Z0-9_+-]+)["\']', source):
        metrics.add(m.group(1))
    # Variable-based: self.variant or name passed to create_metric
    # Look for variant/name = "metric" patterns when create_metric is present
    if "create_metric" in source and "import pyiqa" in source:
        for m in re.finditer(r'(?:variant|name|metric_name)\s*=\s*["\']([a-zA-Z0-9_+-]+)["\']', source):
            metrics.add(m.group(1))
        # Also check default_config for variant
        for m in re.finditer(r'"variant"\s*:\s*["\']([a-zA-Z0-9_+-]+)["\']', source):
            metrics.add(m.group(1))
    return sorted(metrics)


def _extract_torch_hub(source: str) -> List[str]:
    """Extract torch.hub.load() repos."""
    repos = set()
    for m in re.finditer(r'torch\.hub\.load\s*\(\s*["\']([^"\']+)["\']', source):
        repos.add(m.group(1))
    return sorted(repos)


def _extract_torchvision_models(source: str) -> List[str]:
    """Extract torchvision pretrained model usage."""
    models = set()
    # Also catch `models.inception_v3(...)` when `from torchvision import models`
    for m in re.finditer(r'(?:from torchvision\.models\S*\s+import\s+|torchvision\.models\.|models\.)(\w+)', source):
        name = m.group(1)
        if name[0].islower():  # function names like raft_small, resnet18
            models.add(name)
    # Also catch weight enums
    for m in re.finditer(r'(\w+)_Weights\.DEFAULT', source):
        models.add(m.group(1).lower())
    return sorted(models)


def _extract_clip_models(source: str) -> List[str]:
    """Extract CLIP model variants."""
    models = set()
    for m in re.finditer(r'clip\.load\s*\(\s*["\']([^"\']+)["\']', source):
        models.add(m.group(1))
    for m in re.finditer(r'open_clip\.create_model\s*\(\s*["\']([^"\']+)["\']', source):
        models.add(f"open_clip:{m.group(1)}")
    return sorted(models)


def _extract_ffmpeg_models(source: str) -> List[str]:
    """Extract FFmpeg filter/model references."""
    models = set()
    # Only match model version strings used in libvmaf filter args
    for m in re.finditer(r'model=version=(vmaf[_a-z0-9.]+)', source):
        models.add(m.group(1))
    # Also check for explicit model name strings
    for m in re.finditer(r'["\']libvmaf["\']', source):
        models.add("libvmaf")
    if re.search(r'\bxpsnr\b.*filter|filter.*\bxpsnr\b|["\']xpsnr["\']', source):
        models.add("xpsnr")
    if re.search(r'feature=name=cambi', source):
        models.add("cambi")
    if "phone_model=1" in source or "phone=1" in source:
        models.add("vmaf_phone_model")
    return sorted(models)


def _extract_hf_direct_urls(source: str) -> List[tuple]:
    """Extract direct HuggingFace download URLs (resolve/main pattern).

    Also handles Python implicit string concatenation across lines.
    """
    # Pre-process: join adjacent string literals ("a" "b" → "ab")
    joined = re.sub(r'"\s*\n\s*"', '', source)
    results = []
    for m in re.finditer(
        r'https://huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/resolve/main/([^\s"\']+)',
        joined,
    ):
        repo = m.group(1)
        path = m.group(2)
        results.append((repo, path))
    return results


def _extract_required_files(cls) -> Dict[str, str]:
    """Extract required_files class attribute."""
    return getattr(cls, "required_files", {}) or {}


def _generate_charts(
    source_counts: Dict[str, int],
    license_counts: List[tuple],
    vram_tiers: List[tuple],
    top_used: List[tuple],
    output_dir: Path,
) -> Dict[str, str]:
    """Generate PNG charts using seaborn/matplotlib.

    Returns dict of chart_name -> relative path to PNG file.
    """
    paths: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir.mkdir(parents=True, exist_ok=True)
        _colors = ['#6C5CE7', '#00B894', '#FD79A8', '#0984E3', '#FDCB6E',
                    '#E17055', '#00CEC9', '#636E72', '#A29BFE', '#FAB1A0',
                    '#55EFC4', '#DFE6E9', '#74B9FF']

        _W = 5  # uniform width for all charts (paired 2-per-row)

        def _save_bar(items, fname, palette=None):
            sns.set_theme(style="whitegrid", font_scale=0.9)
            labels = [l for l, _ in items]
            values = [v for _, v in items]
            cols = (palette or _colors)[:len(items)]
            fig, ax = plt.subplots(figsize=(_W, max(2.5, len(items) * 0.32)))
            bars = ax.barh(labels[::-1], values[::-1], color=cols[::-1],
                           height=0.65, edgecolor="none")
            ax.bar_label(bars, padding=4, fontsize=9, color="#333")
            ax.set_xlim(0, max(values) * 1.12)
            ax.xaxis.set_visible(False)
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.grid(axis="x", alpha=0.3)
            ax.grid(axis="y", visible=False)
            plt.tight_layout()
            p = output_dir / fname
            plt.savefig(str(p), dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            return f"docs/{p.name}"

        # 1. Sources bar
        src_items = sorted(source_counts.items(), key=lambda x: -x[1])
        paths["sources"] = _save_bar(src_items, "models_sources.png")

        # 2. License bar
        if license_counts:
            lic_colors = []
            for label, _ in license_counts:
                if "Commercial" in label:
                    lic_colors.append("#00B894")
                elif "Non" in label:
                    lic_colors.append("#E17055")
                else:
                    lic_colors.append("#636E72")
            paths["licenses"] = _save_bar(
                license_counts, "models_licenses.png", palette=lic_colors)

        # 3. VRAM tiers bar
        if vram_tiers:
            paths["vram"] = _save_bar(
                vram_tiers, "models_vram.png",
                palette=["#74B9FF"] * len(vram_tiers))

        # 4. Top used models bar
        if top_used:
            paths["top_used"] = _save_bar(
                top_used, "models_top_used.png",
                palette=["#A29BFE"] * len(top_used))

    except ImportError:
        pass  # seaborn/matplotlib not available
    except Exception as exc:
        logger.warning(f"Chart generation failed: {exc}")

    return paths


def _format_params(n: Optional[int]) -> Optional[str]:
    """Format parameter count as human-readable string."""
    if not n:
        return None
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e3:.0f}K"


def _format_downloads(n: Optional[int]) -> Optional[str]:
    """Format download count as human-readable string."""
    if not n:
        return None
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _classify_vram_tier(vram_str: Optional[str]) -> Optional[str]:
    """Classify a VRAM estimate into a tier for charting."""
    if not vram_str or vram_str == "N/A":
        return None
    try:
        val = float(re.search(r"~?([\d.]+)", vram_str).group(1))
        if "GB" in vram_str:
            val *= 1024
        if val <= 200:
            return "< 200 MB"
        if val <= 600:
            return "200-600 MB"
        if val <= 1500:
            return "0.6-1.5 GB"
        if val <= 6000:
            return "1.5-6 GB"
        return "> 6 GB"
    except (AttributeError, ValueError):
        return None


def generate_models_doc(fetch_licenses: bool = True) -> str:
    """Generate MODELS.md content.

    Args:
        fetch_licenses: If True, query HuggingFace API for model licenses.
    """
    ModuleRegistry.discover_modules()
    all_modules = ModuleRegistry.list_modules()

    # Collect all model references
    entries: Dict[str, ModelEntry] = {}  # key -> ModelEntry
    source_counts: Dict[str, int] = defaultdict(int)  # source type -> count

    for mod_name in sorted(all_modules):
        cls = ModuleRegistry.get_module(mod_name)
        if cls is None or cls.name == "unnamed_module":
            continue
        source = _get_source(cls)
        if not source:
            continue

        # HuggingFace
        default_config = getattr(cls, "default_config", {}) or {}
        for hf_id in _extract_hf_models(source, default_config):
            key = f"hf:{hf_id}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(hf_id, (None, None))
                entries[key] = ModelEntry(
                    name=hf_id,
                    source="huggingface",
                    url=f"https://huggingface.co/{hf_id}",
                    size_estimate=disk,
                    vram_estimate=vram,
                    install="pip install transformers",
                )
            entries[key].modules.append(mod_name)

        # pyiqa
        for metric in _extract_pyiqa_metrics(source):
            key = f"pyiqa:{metric}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(f"pyiqa:{metric}", (None, None))
                entries[key] = ModelEntry(
                    name=f"pyiqa/{metric}",
                    source="pyiqa",
                    install="pip install pyiqa",
                    size_estimate=disk,
                    vram_estimate=vram,
                    task=_TASK_DB.get(metric),
                    notes="Auto-downloads on first use",
                )
            entries[key].modules.append(mod_name)

        # torch.hub
        for repo in _extract_torch_hub(source):
            key = f"hub:{repo}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(repo, (None, None))
                entries[key] = ModelEntry(
                    name=repo,
                    source="torch_hub",
                    install="pip install torch",
                    size_estimate=disk,
                    vram_estimate=vram,
                    notes="Auto-downloads via torch.hub",
                )
            entries[key].modules.append(mod_name)

        # torchvision
        for tv_model in _extract_torchvision_models(source):
            key = f"tv:{tv_model}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(f"torchvision:{tv_model}", (None, None))
                entries[key] = ModelEntry(
                    name=f"torchvision/{tv_model}",
                    source="torchvision",
                    install="pip install torchvision",
                    size_estimate=disk,
                    vram_estimate=vram,
                    notes="Bundled with torchvision",
                )
            entries[key].modules.append(mod_name)

        # CLIP
        for clip_model in _extract_clip_models(source):
            key = f"clip:{clip_model}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(f"clip:{clip_model}", (None, None))
                entries[key] = ModelEntry(
                    name=f"CLIP {clip_model}",
                    source="clip",
                    install="pip install clip" if "open_clip" not in clip_model else "pip install open-clip-torch",
                    size_estimate=disk,
                    vram_estimate=vram,
                    notes="Auto-downloads on first use",
                )
            entries[key].modules.append(mod_name)

        # FFmpeg
        for ff_model in _extract_ffmpeg_models(source):
            key = f"ff:{ff_model}"
            if key not in entries:
                disk, vram = _SIZE_DB.get(f"ffmpeg:{ff_model}", ("built-in", "N/A"))
                entries[key] = ModelEntry(
                    name=f"ffmpeg/{ff_model}",
                    source="ffmpeg",
                    install="System: ffmpeg (with libvmaf)",
                    size_estimate=disk,
                    vram_estimate=vram,
                    auto_download=False,
                    notes="Requires FFmpeg compiled with libvmaf",
                )
            entries[key].modules.append(mod_name)

        # Direct HF download URLs (AkaneTendo25/ayase-models etc.)
        # Use full module source to catch module-level URL constants
        full_source = _get_module_source(cls)
        for repo, path in _extract_hf_direct_urls(full_source):
            key = f"hf_file:{repo}/{path}"
            if key not in entries:
                entries[key] = ModelEntry(
                    name=path.split("/")[-1],
                    source="huggingface",
                    url=f"https://huggingface.co/{repo}/resolve/main/{path}",
                    notes=f"From `{repo}` repo",
                )
            if mod_name not in entries[key].modules:
                entries[key].modules.append(mod_name)

        # required_files
        for fname, url in _extract_required_files(cls).items():
            key = f"file:{fname}"
            if key not in entries:
                entries[key] = ModelEntry(
                    name=fname,
                    source="local",
                    url=url,
                    notes="Downloaded by pipeline on first use",
                )
            entries[key].modules.append(mod_name)

        # pip packages with bundled models (piq, stlpips, dreamsim, ptlflow, etc.)
        _PIP_MODEL_PACKAGES = {
            "piq": ("piq", "piq (PyTorch Image Quality)"),
            "stlpips_pytorch": ("stlpips-pytorch", "ST-LPIPS spatiotemporal perceptual"),
            "dreamsim": ("dreamsim", "DreamSim CLIP+DINO similarity"),
            "ptlflow": ("ptlflow", "ptlflow optical flow models"),
            "aesthetic_predictor_v2_5": ("aesthetic-predictor-v2-5", "Aesthetic Predictor V2.5 (SigLIP)"),
            "erqa": ("erqa", "ERQA edge restoration quality"),
            "torchmetrics": ("torchmetrics[audio]", "TorchMetrics (DNSMOS, etc.)"),
            "ultralytics": ("ultralytics", "YOLOv8 object detection"),
            "mediapipe": ("mediapipe", "MediaPipe (face/pose/hand detection)"),
            "deepface": ("deepface", "DeepFace (face recognition/verification)"),
            "insightface": ("insightface", "InsightFace (face recognition)"),
            "onnxruntime": ("onnxruntime", "ONNX Runtime (model inference)"),
            "jxlpy": ("jxlpy", "JPEG XL codec library"),
            "fasttext": ("fasttext", "FastText (text classification)"),
            "joblib": ("joblib", "Joblib (serialized model storage)"),
        }
        # Also check for ultralytics/YOLO in source
        if "yolo" in source.lower() or "ultralytics" in source.lower():
            key = "pip:ultralytics"
            if key not in entries:
                entries[key] = ModelEntry(
                    name="ultralytics", source="pip",
                    install="pip install ultralytics",
                    notes="YOLOv8 object detection",
                )
            if mod_name not in entries[key].modules:
                entries[key].modules.append(mod_name)
        for pkg_import, (pip_name, desc) in _PIP_MODEL_PACKAGES.items():
            if f"import {pkg_import}" in source or f"from {pkg_import}" in source:
                key = f"pip:{pip_name}"
                if key not in entries:
                    entries[key] = ModelEntry(
                        name=pip_name, source="pip",
                        install=f"pip install {pip_name}",
                        notes=desc,
                    )
                if mod_name not in entries[key].modules:
                    entries[key].modules.append(mod_name)

    # Count by source
    for e in entries.values():
        source_counts[e.source] += 1

    # Deduplicate module lists
    for e in entries.values():
        e.modules = sorted(set(e.modules))

    # ── Resolve licenses ─────────────────────────────────────────────────
    # 1. Apply hardcoded known licenses
    for key, entry in entries.items():
        if key in _KNOWN_LICENSES:
            lic, comm = _KNOWN_LICENSES[key]
            entry.license = lic
            entry.commercial_ok = comm

    # 2. Fetch rich HF metadata via API (license, downloads, params, etc.)
    if fetch_licenses:
        hf_repos_to_query = []
        for key, entry in entries.items():
            if entry.source == "huggingface":
                if "/" in entry.name and entry.name.count("/") == 1:
                    hf_repos_to_query.append(entry.name)

        if hf_repos_to_query:
            logger.info("Fetching metadata for %d HuggingFace models...", len(hf_repos_to_query))
            hf_info = _fetch_hf_info_batch(hf_repos_to_query)
            for key, entry in entries.items():
                if entry.source == "huggingface" and entry.name in hf_info:
                    data = hf_info[entry.name]
                    if entry.license is None and data.get("license"):
                        entry.license = data["license"]
                        entry.commercial_ok = _classify_license(data["license"])
                    entry.downloads = data.get("downloads")
                    entry.likes = data.get("likes")
                    entry.parameters = data.get("parameters")
                    entry.pipeline_tag = data.get("pipeline_tag")
                    entry.library = data.get("library")
                    entry.arxiv = data.get("arxiv")
                    # Auto-fill size estimate from parameters if not known
                    if entry.size_estimate is None and entry.parameters:
                        size_mb = entry.parameters * 4 / 1024 / 1024  # FP32
                        if size_mb > 1024:
                            entry.size_estimate = f"~{size_mb/1024:.1f} GB"
                        else:
                            entry.size_estimate = f"~{size_mb:.0f} MB"

    # 3. Inherit license from parent repo for file entries
    # e.g., AkaneTendo25/ayase-models files inherit from the repo
    repo_licenses: Dict[str, tuple] = {}
    for key, entry in entries.items():
        if entry.source == "huggingface" and entry.license and "/" in entry.name and entry.name.count("/") == 1:
            repo_licenses[entry.name] = (entry.license, entry.commercial_ok)
    for key, entry in entries.items():
        if entry.license is None and entry.notes and "From `" in (entry.notes or ""):
            for repo, (lic, comm) in repo_licenses.items():
                if repo in (entry.notes or ""):
                    entry.license = lic
                    entry.commercial_ok = comm
                    break

    # ── Compute chart data ───────────────────────────────────────────────
    total_models = len(entries)

    comm_yes = sum(1 for e in entries.values() if e.commercial_ok is True)
    comm_no = sum(1 for e in entries.values() if e.commercial_ok is False)
    comm_unknown = sum(1 for e in entries.values() if e.commercial_ok is None)

    license_counts = []
    if comm_yes:
        license_counts.append((f"Commercial OK ({comm_yes})", comm_yes))
    if comm_no:
        license_counts.append((f"Non-commercial ({comm_no})", comm_no))
    if comm_unknown:
        license_counts.append((f"Research / Unspecified ({comm_unknown})", comm_unknown))

    # VRAM tiers
    from collections import Counter
    vram_counter: Counter = Counter()
    for e in entries.values():
        tier = _classify_vram_tier(e.vram_estimate)
        if tier:
            vram_counter[tier] += 1
    tier_order = ["< 200 MB", "200-600 MB", "0.6-1.5 GB", "1.5-6 GB", "> 6 GB"]
    vram_tiers = [(t, vram_counter[t]) for t in tier_order if vram_counter[t]]

    # Top used models (most modules)
    top_used_entries = sorted(entries.values(), key=lambda e: -len(e.modules))[:12]
    top_used = [(e.name.split("/")[-1] if "/" in e.name else e.name, len(e.modules))
                for e in top_used_entries if len(e.modules) > 1]

    docs_dir = Path(__file__).parent.parent.parent / "docs"
    chart_paths = _generate_charts(
        dict(source_counts), license_counts, vram_tiers, top_used, docs_dir,
    )

    # ══════════════════════════════════════════════════════════════════════
    # BUILD DOCUMENT
    # ══════════════════════════════════════════════════════════════════════
    L: List[str] = []
    a = L.append

    from datetime import datetime
    try:
        from ayase import __version__
    except ImportError:
        __version__ = "dev"

    a("# Ayase Models Reference")
    a("")
    a(f"> **Version {__version__}** · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
      f"· **{total_models} models** across **{len(source_counts)} sources**")
    a(">")
    a("> `ayase modules models -o MODELS.md` to regenerate")

    # ── Summary ──────────────────────────────────────────────────────────
    a("")
    a("## Summary")
    a("")
    a(f"**{total_models}** models · **{source_counts.get('huggingface', 0)}** HuggingFace "
      f"· **{source_counts.get('pyiqa', 0)}** pyiqa · **{len(source_counts)}** sources")

    # ── Charts (2-per-row) ───────────────────────────────────────────────
    chart_titles = {
        "sources": "Models by Source",
        "licenses": "License Distribution",
        "vram": "VRAM Tiers",
        "top_used": "Top Used Models",
    }
    chart_order = [k for k in ("sources", "licenses", "vram", "top_used") if k in chart_paths]
    for i in range(0, len(chart_order), 2):
        pair = chart_order[i:i + 2]
        if len(pair) == 2:
            t1, t2 = chart_titles[pair[0]], chart_titles[pair[1]]
            p1, p2 = chart_paths[pair[0]], chart_paths[pair[1]]
            a("")
            a('<table width="100%"><tr>')
            a(f'<td width="50%" valign="top"><h4>{t1}</h4><img src="{p1}" width="100%"/></td>')
            a(f'<td width="50%" valign="top"><h4>{t2}</h4><img src="{p2}" width="100%"/></td>')
            a("</tr></table>")
        else:
            a("")
            a(f"<h4>{chart_titles[pair[0]]}</h4>")
            a("")
            a(f"![]({chart_paths[pair[0]]})")

    # Estimated total download size
    known_sizes = []
    for e in entries.values():
        if e.size_estimate and e.size_estimate not in ("built-in", "N/A"):
            try:
                val = float(re.search(r"~?([\d.]+)", e.size_estimate).group(1))
                unit = e.size_estimate.strip()
                if "GB" in unit:
                    val *= 1024
                known_sizes.append(val)
            except (AttributeError, ValueError):
                pass
    if known_sizes:
        total_mb = sum(known_sizes)
        a("")
        if total_mb > 1024:
            a(f"**Estimated total download size (all models):** ~{total_mb / 1024:.0f} GB")
        else:
            a(f"**Estimated total download size (all models):** ~{total_mb:.0f} MB")
        a("")
        a("*Note: Most modules auto-download only the models they need on first use. "
          "You rarely need all models at once.*")

    # ── License warning ──────────────────────────────────────────────────
    comm_research = sum(1 for e in entries.values()
                        if e.license and "research" in (e.license or "").lower()
                        and e.commercial_ok is None)
    if comm_no > 0 or comm_research > 0:
        a("")
        a("> [!WARNING]")
        a("> **Commercial use:** Stick to modules whose models are marked "
          "\"Commercial OK\" above. Most pyiqa metrics marked \"research\" "
          "are re-implementations under pyiqa's MIT license, but the original "
          "training data or architecture may carry restrictions — verify before "
          "commercial deployment.")

    # ── Validate URLs (stderr warnings only) ────────────────────────────
    if fetch_licenses:
        url_warnings = _validate_urls(entries)
        # Filter out 401 (gated/private repos — expected for some HF models)
        real_broken = [w for w in url_warnings if "HTTP 401" not in w]
        gated = [w for w in url_warnings if "HTTP 401" in w]
        import sys
        if real_broken:
            print(f"WARNING: {len(real_broken)} broken model URL(s):", file=sys.stderr)
            for w in real_broken:
                print(w, file=sys.stderr)
        if gated:
            print(f"INFO: {len(gated)} gated/private model(s) (auth required):", file=sys.stderr)
            for w in gated:
                print(w, file=sys.stderr)

    # ── Category navigation ──────────────────────────────────────────────
    # Prepare section data for navigation and rendering
    hf_entries = [(k, e) for k, e in sorted(entries.items())
                  if e.source == "huggingface" and not (e.notes and "From `" in (e.notes or ""))]
    hf_file_entries = [(k, e) for k, e in sorted(entries.items())
                       if e.source == "huggingface" and e.notes and "From `" in (e.notes or "")]
    pyiqa_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "pyiqa"]
    tv_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "torchvision"]
    clip_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "clip"]
    hub_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "torch_hub"]
    ff_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "ffmpeg"]
    pip_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "pip"]

    # Group weight files by parent repo
    weight_file_repos: Dict[str, List[tuple]] = defaultdict(list)
    for _key, e in hf_file_entries:
        # Extract repo name from notes "From `repo` repo"
        m = re.search(r"From `([^`]+)` repo", e.notes or "")
        if m:
            weight_file_repos[m.group(1)].append(e)

    nav_sections = []
    if hf_entries:
        nav_sections.append((f"HuggingFace ({len(hf_entries)})", "huggingface-models"))
    if weight_file_repos:
        total_wf = sum(len(v) for v in weight_file_repos.values())
        nav_sections.append((f"Weight Files ({total_wf})", "weight-file-repos"))
    if pyiqa_entries:
        nav_sections.append((f"pyiqa ({len(pyiqa_entries)})", "pyiqa-metrics"))
    if tv_entries:
        nav_sections.append((f"torchvision ({len(tv_entries)})", "torchvision-models"))
    if clip_entries:
        nav_sections.append((f"CLIP / OpenCLIP ({len(clip_entries)})", "clip--openclip"))
    if hub_entries:
        nav_sections.append((f"torch.hub ({len(hub_entries)})", "torchhub"))
    if ff_entries:
        nav_sections.append((f"FFmpeg ({len(ff_entries)})", "ffmpeg"))
    if pip_entries:
        nav_sections.append((f"pip Packages ({len(pip_entries)})", "pip-packages"))
    nav_sections.append(("Quick Install Guide", "quick-install-guide"))

    a("")
    a('<a id="categories"></a>')
    a("")
    a(" · ".join(f"[{label}](#{anchor})" for label, anchor in nav_sections))
    a("")
    a("---")
    a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: HuggingFace Models
    # ══════════════════════════════════════════════════════════════════════
    if hf_entries:
        a(f"## HuggingFace Models")
        a("")

        for _key, e in hf_entries:
            # Heading with link
            if e.url:
                a(f'### <a href="{e.url}" target="_blank">`{e.name}`</a> [↑](#categories)')
            else:
                a(f"### `{e.name}` [↑](#categories)")

            # Tagline: pipeline_tag + license
            tagline_parts = []
            if e.pipeline_tag:
                tagline_parts.append(e.pipeline_tag)
            if e.license:
                tagline_parts.append(e.license)
            if tagline_parts:
                a(f"> {' · '.join(tagline_parts)}")
            a("")

            # Used by
            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")

            # Parameters + Downloads
            info_parts = []
            params = _format_params(e.parameters)
            if params:
                info_parts.append(f"**Parameters**: {params}")
            dl = _format_downloads(e.downloads)
            if dl:
                info_parts.append(f"**Downloads**: {dl}")
            if info_parts:
                a(f"- {' · '.join(info_parts)}")

            # VRAM + Disk
            size_parts = []
            if e.vram_estimate:
                size_parts.append(f"**VRAM**: {e.vram_estimate}")
            if e.size_estimate:
                size_parts.append(f"**Disk**: {e.size_estimate}")
            if size_parts:
                a(f"- {' · '.join(size_parts)}")

            # Source link (arXiv)
            if e.arxiv:
                a(f'- **Source**: <a href="https://arxiv.org/abs/{e.arxiv}" target="_blank">arXiv</a>')

            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Weight File Repos (grouped)
    # ══════════════════════════════════════════════════════════════════════
    if weight_file_repos:
        a(f"## Weight File Repos")
        a("")

        for repo_name, files in sorted(weight_file_repos.items()):
            repo_url = f"https://huggingface.co/{repo_name}"
            a(f'### <a href="{repo_url}" target="_blank">`{repo_name}`</a> [↑](#categories)')
            a("> Pre-trained weight files for ayase modules")
            a("")
            for fe in sorted(files, key=lambda x: x.name):
                file_mods = ", ".join(f"`{m}`" for m in fe.modules)
                a(f"- `{fe.name}` — used by {file_mods}")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: pyiqa Metrics
    # ══════════════════════════════════════════════════════════════════════
    if pyiqa_entries:
        a(f"## pyiqa Metrics ({len(pyiqa_entries)})")
        a("")
        a('<a href="https://github.com/chaofengc/IQA-PyTorch" target="_blank">pyiqa</a> is an MIT-licensed collection '
          "of image/video quality metrics. Weights auto-download on first "
          "`pyiqa.create_metric()` call. `pip install pyiqa`")
        a("")
        a("| Metric | Task | License | Commercial | Used By |")
        a("|--------|------|---------|------------|---------|")
        for _key, e in pyiqa_entries:
            metric_name = e.name.replace("pyiqa/", "")
            task = e.task or "IQA"
            lic = e.license or "research"
            comm = {True: "Yes", False: "No"}.get(e.commercial_ok, "—")
            mods = ", ".join(f"`{m}`" for m in e.modules[:4])
            if len(e.modules) > 4:
                mods += f" +{len(e.modules) - 4}"
            a(f"| `{metric_name}` | {task} | {lic} | {comm} | {mods} |")
        a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: torchvision Models
    # ══════════════════════════════════════════════════════════════════════
    if tv_entries:
        a(f"## torchvision Models")
        a("")
        a("Bundled with `pip install torchvision`. Weights download on first use.")
        a("")

        for _key, e in tv_entries:
            a(f"### `{e.name}` [↑](#categories)")

            tagline_parts = ["torchvision"]
            lic_key = f"tv:{e.name.split('/')[-1]}"
            lic_info = _KNOWN_LICENSES.get(lic_key)
            if lic_info:
                tagline_parts.append(lic_info[0])
            a(f"> {' · '.join(tagline_parts)}")
            a("")

            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")

            size_parts = []
            if e.vram_estimate:
                size_parts.append(f"**VRAM**: {e.vram_estimate}")
            if e.size_estimate:
                size_parts.append(f"**Disk**: {e.size_estimate}")
            if size_parts:
                a(f"- {' · '.join(size_parts)}")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: CLIP / OpenCLIP
    # ══════════════════════════════════════════════════════════════════════
    if clip_entries:
        a(f"## CLIP / OpenCLIP")
        a("")

        for _key, e in clip_entries:
            a(f"### `{e.name}` [↑](#categories)")

            tagline_parts = []
            lic_key = f"clip:{e.name.replace('CLIP ', '')}"
            lic_info = _KNOWN_LICENSES.get(lic_key)
            if lic_info:
                tagline_parts.append(lic_info[0])
            if tagline_parts:
                a(f"> {' · '.join(tagline_parts)}")
            a("")

            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")

            size_parts = []
            if e.vram_estimate:
                size_parts.append(f"**VRAM**: {e.vram_estimate}")
            if e.size_estimate:
                size_parts.append(f"**Disk**: {e.size_estimate}")
            if size_parts:
                a(f"- {' · '.join(size_parts)}")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: torch.hub
    # ══════════════════════════════════════════════════════════════════════
    if hub_entries:
        a(f"## torch.hub")
        a("")

        for _key, e in hub_entries:
            a(f"### `{e.name}` [↑](#categories)")

            tagline_parts = ["torch.hub"]
            lic_key = f"hub:{e.name}"
            lic_info = _KNOWN_LICENSES.get(lic_key)
            if lic_info:
                tagline_parts.append(lic_info[0])
            a(f"> {' · '.join(tagline_parts)}")
            a("")

            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")

            size_parts = []
            if e.vram_estimate:
                size_parts.append(f"**VRAM**: {e.vram_estimate}")
            if e.size_estimate:
                size_parts.append(f"**Disk**: {e.size_estimate}")
            if size_parts:
                a(f"- {' · '.join(size_parts)}")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: FFmpeg
    # ══════════════════════════════════════════════════════════════════════
    if ff_entries:
        a(f"## FFmpeg")
        a("")
        a("Require FFmpeg compiled with libvmaf. No separate download needed.")
        a("")

        for _key, e in ff_entries:
            a(f"### `{e.name}` [↑](#categories)")

            tagline_parts = ["built-in"]
            lic_key = f"ff:{e.name.split('/')[-1]}"
            lic_info = _KNOWN_LICENSES.get(lic_key)
            if lic_info:
                tagline_parts.append(lic_info[0])
            a(f"> {' · '.join(tagline_parts)}")
            a("")

            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: pip Packages
    # ══════════════════════════════════════════════════════════════════════
    if pip_entries:
        a(f"## pip Packages")
        a("")

        for _key, e in pip_entries:
            a(f"### `{e.name}` [↑](#categories)")
            if e.notes:
                a(f"> {e.notes}")
            a("")

            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"- **Used by**: {mods}")
            if e.install:
                a(f"- **Install**: `{e.install}`")
            a("")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION: Quick Install Guide
    # ══════════════════════════════════════════════════════════════════════
    a("## Quick Install Guide")
    a("")
    a("Install all model dependencies at once:")
    a("")
    a("```bash")
    a("# Core (covers ~80% of modules)")
    a("pip install torch torchvision pyiqa piq opencv-python Pillow transformers")
    a("")
    a("# Audio metrics")
    a("pip install librosa soundfile pesq pystoi")
    a("")
    a("# Additional NR/FR metrics")
    a("pip install lpips dreamsim ssimulacra2 stlpips-pytorch")
    a("")
    a("# Video-specific")
    a("pip install decord scenedetect")
    a("")
    a("# Optional heavy models (LLaVA, Q-Align)")
    a("pip install accelerate bitsandbytes  # for efficient LLM loading")
    a("```")

    a("")
    return "\n".join(L)
