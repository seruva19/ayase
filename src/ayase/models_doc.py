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
from typing import Dict, List, Optional, Set

from .pipeline import ModuleRegistry

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
    "brisque": "No-reference image quality (naturalness)",
    "niqe": "No-reference image quality (naturalness statistics)",
    "clipiqa+": "CLIP-based image quality assessment",
    "maniqa": "Multi-dimension attention NR-IQA",
    "topiq_nr": "Transformer-based NR image quality",
    "topiq_fr": "Transformer-based FR image quality",
    "dbcnn": "Deep bilinear CNN for blind IQA",
    "musiq": "Multi-scale image quality transformer",
    "nima": "Neural image assessment (aesthetic + technical)",
    "hyperiqa": "Adaptive hypernetwork NR image quality",
    "ahiq": "Attention-based hybrid FR-IQA",
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
        # Skip direct file URLs (not model repos)
        if "/" in model_id and model_id.count("/") == 1:
            lic = _fetch_hf_license(model_id)
            if lic:
                result[model_id] = lic
                logger.info("  %s → %s", model_id, lic)
            else:
                logger.info("  %s → (no license found)", model_id)
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
    # Direct from_pretrained calls
    for m in re.finditer(r'from_pretrained\s*\(\s*["\']([a-zA-Z0-9_/-]+)["\']', source):
        candidate = m.group(1)
        if "/" in candidate and not any(x in candidate for x in ("http", "path", ".py")):
            models.add(candidate)
    # Model names from default_config (model_name, vlm_model, etc.)
    if default_config:
        for key in ("model_name", "vlm_model", "vlm_model_name"):
            val = default_config.get(key, "")
            if isinstance(val, str) and "/" in val and not val.startswith(("http", "/")):
                models.add(val)
    return sorted(models)


def _extract_pyiqa_metrics(source: str) -> List[str]:
    """Extract pyiqa.create_metric() metric names."""
    metrics = set()
    for m in re.finditer(r'create_metric\s*\(\s*["\']([a-zA-Z0-9_+-]+)["\']', source):
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
    for m in re.finditer(r'(?:from torchvision\.models\S*\s+import\s+|torchvision\.models\.)(\w+)', source):
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
    """Extract direct HuggingFace download URLs (resolve/main pattern)."""
    results = []
    for m in re.finditer(
        r'https://huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/resolve/main/([^\s"\']+)',
        source,
    ):
        repo = m.group(1)
        path = m.group(2)
        results.append((repo, path))
    return results


def _extract_required_files(cls) -> Dict[str, str]:
    """Extract required_files class attribute."""
    return getattr(cls, "required_files", {}) or {}


def _bar_chart(items, max_width=30):
    if not items:
        return []
    max_val = max(v for _, v in items)
    max_label = max(len(str(lb)) for lb, _ in items)
    lines = []
    for label, val in items:
        bar_len = int(val / max(max_val, 1) * max_width)
        bar = "█" * bar_len
        lines.append(f"  {str(label):<{max_label}}  {bar} {val}")
    return lines


def generate_models_doc(fetch_licenses: bool = True) -> str:
    """Generate MODELS.md content.

    Args:
        fetch_licenses: If True, query HuggingFace API for model licenses.
    """
    all_modules = ModuleRegistry.list_modules()

    # Collect all model references
    entries: Dict[str, ModelEntry] = {}  # key → ModelEntry
    source_counts = defaultdict(int)  # source type → count

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

    # 2. Fetch HF licenses via API
    if fetch_licenses:
        hf_repos_to_query = []
        for key, entry in entries.items():
            if entry.source == "huggingface" and entry.license is None:
                # Only query actual repos (org/model), not file entries
                if "/" in entry.name and entry.name.count("/") == 1:
                    hf_repos_to_query.append(entry.name)

        if hf_repos_to_query:
            logger.info("Fetching licenses for %d HuggingFace models...", len(hf_repos_to_query))
            hf_licenses = _fetch_hf_licenses_batch(hf_repos_to_query)
            for key, entry in entries.items():
                if entry.source == "huggingface" and entry.license is None:
                    lic = hf_licenses.get(entry.name)
                    if lic:
                        entry.license = lic
                        entry.commercial_ok = _classify_license(lic)

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

    # ── Build document ───────────────────────────────────────────────────
    L = []
    a = L.append

    a("# Ayase Models Reference")
    a("")
    a("Auto-generated catalog of all ML models, weights, and external assets "
      "used by Ayase pipeline modules. Run `ayase modules models` to regenerate.")

    # Summary
    a("")
    a("## Summary")
    a("")
    total_models = len(entries)
    total_hf = source_counts.get("huggingface", 0)
    total_pyiqa = source_counts.get("pyiqa", 0)
    a("| Stat | Value |")
    a("|------|-------|")
    a(f"| Total model references | **{total_models}** |")
    a(f"| HuggingFace models | {total_hf} |")
    a(f"| pyiqa metrics | {total_pyiqa} |")
    a(f"| torchvision models | {source_counts.get('torchvision', 0)} |")
    a(f"| CLIP variants | {source_counts.get('clip', 0)} |")
    a(f"| torch.hub repos | {source_counts.get('torch_hub', 0)} |")
    a(f"| FFmpeg models | {source_counts.get('ffmpeg', 0)} |")
    a(f"| Local weight files | {source_counts.get('local', 0)} |")

    # Chart
    a("")
    a("### By Source")
    a("")
    a("```")
    for line in _bar_chart(sorted(source_counts.items(), key=lambda x: -x[1])):
        a(line)
    a("```")

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
            a(f"**Estimated total download size (all models):** ~{total_mb/1024:.0f} GB")
        else:
            a(f"**Estimated total download size (all models):** ~{total_mb:.0f} MB")
        a("")
        a("*Note: Most modules auto-download only the models they need on first use. "
          "You rarely need all models at once.*")

    # ── License Summary ────────────────────────────────────────────────
    comm_yes = sum(1 for e in entries.values() if e.commercial_ok is True)
    comm_no = sum(1 for e in entries.values() if e.commercial_ok is False)
    comm_unknown = sum(1 for e in entries.values() if e.commercial_ok is None)
    comm_research = sum(1 for e in entries.values()
                        if e.license and "research" in (e.license or "").lower()
                        and e.commercial_ok is None)

    a("")
    a("### License Overview")
    a("")
    a("```")
    lic_items = [
        ("Commercial OK", comm_yes),
        ("Non-commercial", comm_no),
        ("Research / unspecified", comm_unknown),
    ]
    for line in _bar_chart(lic_items):
        a(line)
    a("```")
    a("")
    if comm_no > 0 or comm_research > 0:
        a("> **For commercial use:** Stick to modules whose models are marked "
          "\"Commercial: Yes\" below. Most pyiqa metrics marked \"research\" "
          "are re-implementations under pyiqa's MIT license, but the original "
          "training data or architecture may carry restrictions — verify before "
          "commercial deployment.")
        a("")

    # ── HuggingFace Models ───────────────────────────────────────────────
    hf_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "huggingface"]
    if hf_entries:
        a("")
        a("## HuggingFace Models")
        a("")
        a("| Model | License | Commercial | Disk | VRAM | Used By |")
        a("|-------|---------|------------|------|------|---------|")
        for _key, e in hf_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules[:5])
            if len(e.modules) > 5:
                mods += f" +{len(e.modules)-5}"
            disk = e.size_estimate or "?"
            vram = e.vram_estimate or "?"
            lic = e.license or "?"
            comm = {True: "Yes", False: "No", None: "?"}[e.commercial_ok]
            link = f"[{e.name}]({e.url})" if e.url else e.name
            a(f"| {link} | {lic} | {comm} | {disk} | {vram} | {mods} |")

    # ── pyiqa Models ─────────────────────────────────────────────────────
    pyiqa_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "pyiqa"]
    if pyiqa_entries:
        a("")
        a("## pyiqa Metrics")
        a("")
        a("All auto-download weights on first `pyiqa.create_metric()` call. "
          "pyiqa itself is MIT-licensed; underlying model licenses vary.")
        a("")
        a("| Metric | License | Commercial | Task | VRAM | Used By |")
        a("|--------|---------|------------|------|------|---------|")
        for _key, e in pyiqa_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules[:4])
            if len(e.modules) > 4:
                mods += f" +{len(e.modules)-4}"
            vram = e.vram_estimate or "~200 MB"
            task = e.task or "IQA"
            lic = e.license or "research"
            comm = {True: "Yes", False: "No", None: "?"}[e.commercial_ok]
            a(f"| `{e.name}` | {lic} | {comm} | {task} | {vram} | {mods} |")

    # ── torchvision Models ───────────────────────────────────────────────
    tv_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "torchvision"]
    if tv_entries:
        a("")
        a("## torchvision Models")
        a("")
        a("Bundled with `pip install torchvision`. Weights download on first use.")
        a("")
        a("| Model | Disk | VRAM | Used By |")
        a("|-------|------|------|---------|")
        for _key, e in tv_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules[:5])
            disk = e.size_estimate or "?"
            vram = e.vram_estimate or "?"
            a(f"| `{e.name}` | {disk} | {vram} | {mods} |")

    # ── CLIP Models ──────────────────────────────────────────────────────
    clip_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "clip"]
    if clip_entries:
        a("")
        a("## CLIP Models")
        a("")
        a("| Model | Disk | VRAM | Used By |")
        a("|-------|------|------|---------|")
        for _key, e in clip_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules[:5])
            if len(e.modules) > 5:
                mods += f" +{len(e.modules)-5}"
            disk = e.size_estimate or "?"
            vram = e.vram_estimate or "?"
            a(f"| `{e.name}` | {disk} | {vram} | {mods} |")

    # ── torch.hub Models ────────────────────────────────────────────────
    hub_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "torch_hub"]
    if hub_entries:
        a("")
        a("## torch.hub Models")
        a("")
        a("| Repo | Disk | VRAM | Used By |")
        a("|------|------|------|---------|")
        for _key, e in hub_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules)
            disk = e.size_estimate or "?"
            vram = e.vram_estimate or "?"
            a(f"| `{e.name}` | {disk} | {vram} | {mods} |")

    # ── FFmpeg Models ────────────────────────────────────────────────────
    ff_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "ffmpeg"]
    if ff_entries:
        a("")
        a("## FFmpeg Models")
        a("")
        a("Require FFmpeg compiled with libvmaf. No separate download needed.")
        a("")
        a("| Model | Used By |")
        a("|-------|---------|")
        for _key, e in ff_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules)
            a(f"| `{e.name}` | {mods} |")

    # ── Local/Downloaded Files ───────────────────────────────────────────
    local_entries = [(k, e) for k, e in sorted(entries.items()) if e.source == "local"]
    if local_entries:
        a("")
        a("## Local Weight Files")
        a("")
        a("Downloaded by the pipeline on first use via `required_files`.")
        a("")
        a("| File | URL | Used By |")
        a("|------|-----|---------|")
        for _key, e in local_entries:
            mods = ", ".join(f"`{m}`" for m in e.modules)
            url = e.url or "—"
            a(f"| `{e.name}` | {url} | {mods} |")

    # ── Quick Install Guide ──────────────────────────────────────────────
    a("")
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
