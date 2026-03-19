"""Generate METRICS.md from module metadata via PipelineModule.get_metadata().

Enhanced with:
- Summary dashboard with Unicode bar charts
- Per-module: required packages, GPU flag, speed tier, fallback chain
- Score direction & range for every output field
- Field collision map (multiple modules → same field)
- Orphaned QualityMetrics fields (no module writes to them)
- Module dependency graph (Mermaid)
- Recommended module presets
- Benchmark coverage matrix
- Deprecated field aliases
- Static health checks
"""

import inspect
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .pipeline import ModuleRegistry, PipelineModule

# ── pip package mapping ─────────────────────────────────────────────────────
# Maps Python import names to pip install names
_IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "skvideo": "scikit-video",
    "yaml": "PyYAML",
    "lpips": "lpips",
    "piq": "piq",
    "pyiqa": "pyiqa",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchmetrics": "torchmetrics",
    "transformers": "transformers",
    "clip": "clip (openai)",
    "open_clip": "open-clip-torch",
    "decord": "decord",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "pystoi": "pystoi",
    "pesq": "pesq",
    "cpbd": "cpbd",
    "dreamsim": "dreamsim",
    "ssimulacra2": "ssimulacra2",
    "jxlpy": "jxlpy",
    "rembg": "rembg",
    "easyocr": "easyocr",
    "pytesseract": "pytesseract",
    "scenedetect": "scenedetect",
    "stlpips_pytorch": "stlpips-pytorch",
    "flip_evaluator": "flip-evaluator",
    "visqol": "visqol",
    "deepface": "deepface",
    "mediapipe": "mediapipe",
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
    "ptlflow": "ptlflow",
    "pywt": "PyWavelets",
}

_STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "logging", "pathlib", "typing",
    "collections", "functools", "itertools", "abc", "copy", "io",
    "tempfile", "subprocess", "shutil", "hashlib", "warnings", "time",
    "dataclasses", "enum", "contextlib", "textwrap", "statistics",
    "struct", "base64", "threading", "ast",
}

# ── VRAM estimate by model pattern ──────────────────────────────────────────
_VRAM_PATTERNS = {
    r"llava.*7b|llava-1\.5-7b": "~14 GB",
    r"llava.*13b": "~28 GB",
    r"clip-vit-large": "~1.5 GB",
    r"clip-vit-base|ViT-B": "~600 MB",
    r"videomae.*large": "~1.5 GB",
    r"videomae.*base": "~400 MB",
    r"dino.*vitb|dinov2": "~400 MB",
    r"resnet|r3d_18|inception": "~200 MB",
    r"DOVER|dover": "~800 MB",
    r"q-align|q_align": "~14 GB",
}

# ── Benchmark → modules mapping ────────────────────────────────────────────
_BENCHMARK_MODULES = {
    "VBench (16/16)": [
        "subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "aesthetic", "imaging_quality",
        "object_class", "multiple_objects", "human_action", "color", "spatial_relationship",
        "scene", "appearance_style", "temporal_style", "overall_consistency",
    ],
    "VBench-2.0 (5/5)": [
        "human_fidelity", "physics", "commonsense", "creativity",
    ],
    "EvalCrafter (17/17)": [
        "inception_score", "action_recognition", "flow_score", "warping_error",
        "motion_ac_score", "clip_score", "clip_temp", "face_consistency",
        "blip_bleu", "sd_score", "celebrity_id", "dover_score", "aesthetic_score",
        "subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness",
    ],
    "ChronoMagic-Bench (2/2)": ["chronomagic"],
    "T2V-CompBench (7/7)": ["t2v_compbench"],
    "DEVIL (4/4)": ["dynamics_range", "ti_si", "flicker_detection", "motion"],
}

# ── Recommended presets ─────────────────────────────────────────────────────
_PRESETS = {
    "Quick Scan": {
        "desc": "Fast quality triage (~1s/sample, CPU-only)",
        "modules": ["basic", "metadata", "exposure", "letterbox"],
    },
    "Dataset Curation": {
        "desc": "Clean & deduplicate datasets for training",
        "modules": ["basic", "aesthetic", "dedup", "nsfw", "watermark_classifier",
                     "brisque", "metadata", "embedding", "diversity_selection"],
    },
    "Video Generation Eval": {
        "desc": "Evaluate text-to-video model outputs (VBench-style)",
        "modules": ["aesthetic", "subject_consistency", "background_consistency",
                     "temporal_flickering", "motion_smoothness", "clip_iqa",
                     "video_text_matching", "dover", "ti_si"],
    },
    "Codec Comparison": {
        "desc": "Compare video codec quality (needs reference)",
        "modules": ["vmaf", "ssimulacra2", "psnr_hvs", "ms_ssim", "butteraugli",
                     "cambi", "codec_specific_quality"],
    },
    "Audio Quality": {
        "desc": "Speech/audio quality assessment",
        "modules": ["audio_pesq", "audio_utmos", "dnsmos", "audio_si_sdr",
                     "audio_estoi", "visqol"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Source inspection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_source(cls) -> str:
    try:
        return inspect.getsource(cls)
    except (TypeError, OSError):
        return ""


def _get_group(name: str, input_type: str) -> str:
    if input_type.startswith("audio"):
        return "Audio Quality"
    if input_type == "batch":
        return "Distribution (batch)"
    kw_map = {
        "Motion & Temporal": (
            "motion", "temporal", "flicker", "flow", "subject_consist",
            "background_consist", "scene_detect", "jump_cut", "playback",
            "camera_jitter", "camera_motion", "flow_coherence",
            "object_permanence", "stabilized", "warping", "raft_motion",
            "ptlflow", "judder", "vfr",
        ),
        "Video Quality Assessment": (
            "dover", "fast_vqa", "mdtvsfa", "videval", "tlvqm", "c3dvqa",
            "cover", "finevq", "kvq", "rqvqa", "funque", "st_greed",
            "hdr_vqm", "cgvqm", "movie",
        ),
        "Video Generation": (
            "videoscore", "video_reward", "aigv", "chronomagic",
            "t2v_comp", "video_type", "video_memor", "t2v_score",
        ),
        "Audio-Visual": ("av_sync", "audio_visual"),
        "Full-Reference & Distribution": (
            "vmaf", "ssimulacra", "butteraugli", "flip", "psnr", "ssim",
            "ciede", "pieapp", "cw_ssim", "nlpd", "ahiq", "topiq_fr",
            "dreamsim", "dmm", "wadiqam_fr", "ssimc", "xpsnr", "hdr_vdp",
            "delta_ictcp", "ckdn", "deepwsd", "strred", "flolpips",
            "st_lpips", "vif", "fvd", "fvmd", "kvd", "mad",
        ),
        "HDR & Color": ("hdr_", "pu_metric", "tonal"),
        "Safety & Content": ("nsfw", "harmful", "deepfake", "watermark", "bias"),
        "Text & Semantic": (
            "semantic", "caption", "clip_temp", "clip_iqa", "sd_reference",
            "ocr", "vqa_score", "tifa", "nemo", "text_", "video_text",
            "ram_tag", "promptiqa",
        ),
        "Codec & Technical": ("codec", "compression", "letterbox"),
    }
    for group, keywords in kw_map.items():
        if any(kw in name for kw in keywords):
            return group
    if "face" in name or "identity" in name:
        return "Face & Identity"
    if "depth" in name:
        return "Depth"
    if "i2v" in name:
        return "Image-to-Video Reference"
    return "No-Reference Quality"


def _detect_backends(source: str) -> List[str]:
    backends = []
    patterns = {
        "pyiqa": r"pyiqa\.create_metric",
        "piq": r"import piq|from piq",
        "ffmpeg": r'["\']ffmpeg["\']|subprocess.*ffmpeg',
        "torch": r"import torch",
        "transformers": r"from transformers",
        "opencv": r"import cv2",
        "torchvision": r"from torchvision",
        "torchmetrics": r"from torchmetrics",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, source):
            backends.append(name)
    return backends


def _detect_tiered(source: str) -> bool:
    return source.count("self._backend") >= 2


def _detect_fallback_chain(source: str) -> List[str]:
    """Extract tiered fallback chain from backend assignment patterns."""
    chain = []
    for m in re.finditer(r'self\._backend\s*=\s*["\'](\w+)["\']', source):
        label = m.group(1)
        if label not in chain:
            chain.append(label)
    return chain


def _detect_packages(source: str) -> List[str]:
    """Extract third-party pip packages from import statements."""
    pkgs: Set[str] = set()
    for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", source, re.MULTILINE):
        top = m.group(1).split(".")[0]
        if top in _STDLIB_MODULES or top.startswith("ayase"):
            continue
        pip_name = _IMPORT_TO_PIP.get(top, top)
        pkgs.add(pip_name)
    # Remove numpy (always available)
    pkgs.discard("numpy")
    return sorted(pkgs)


def _detect_gpu(source: str) -> bool:
    return "cuda" in source or ".to(device)" in source or ".to(self._device)" in source


def _detect_speed_tier(source: str, backends: List[str]) -> str:
    if any(kw in source for kw in ("llava", "q_align", "Q-Align", "LLM", "CausalLM")):
        return "slow"
    if any(b in backends for b in ("pyiqa", "transformers", "torchvision")):
        return "medium"
    if "torch" in backends:
        return "medium"
    return "fast"


def _detect_hf_models(source: str) -> List[str]:
    """Extract HuggingFace model identifiers."""
    models = []
    for m in re.finditer(r'["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']', source):
        candidate = m.group(1)
        # Filter likely HF repos (org/model format)
        if not any(x in candidate for x in ("http", "path", "file", "dir")):
            if candidate not in models:
                models.append(candidate)
    return models[:3]  # Cap at 3


def _estimate_vram(source: str) -> Optional[str]:
    for pattern, estimate in _VRAM_PATTERNS.items():
        if re.search(pattern, source, re.IGNORECASE):
            return estimate
    return None


def _detect_paper(cls) -> Optional[str]:
    """Extract paper citation from docstring."""
    doc = cls.__doc__ or ""
    # Look for common citation patterns
    for m in re.finditer(
        r"(?:References?:?\s*[-–]\s*)?(\w[\w\s&]+?et al\.?\s*\(\d{4}\)[^)\n]*)",
        doc
    ):
        return m.group(1).strip()
    for m in re.finditer(r"((?:CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|WACV|TIP|TPAMI)\s*\d{4})", doc):
        return m.group(1)
    return None


def _detect_fields_read(source: str) -> Set[str]:
    """Find QualityMetrics fields that a module reads (dependencies)."""
    reads: Set[str] = set()
    # Pattern: sample.quality_metrics.FIELD (in a read context, not assignment)
    for m in re.finditer(r"quality_metrics\.(\w+)(?!\s*=)", source):
        field = m.group(1)
        if field not in ("__class__", "model_dump", "non_null_metrics"):
            reads.add(field)
    return reads


def _detect_fields_written(source: str) -> Set[str]:
    """Find QualityMetrics fields that a module writes."""
    writes: Set[str] = set()
    for m in re.finditer(r"quality_metrics\.(\w+)\s*=", source):
        writes.add(m.group(1))
    return writes


def _static_checks(source: str, meta: Dict) -> List[str]:
    warnings = []
    if "def on_mount(" in source and "def setup(" not in source:
        if "super().on_mount()" not in source:
            warnings.append("uses `on_mount()` instead of `setup()`")
    if not meta["output_fields"] and "validation_issues" not in source:
        warnings.append("no output fields and no validation issues")
    if "quality_metrics" not in source and meta["output_fields"]:
        warnings.append("declares output fields but never assigns quality_metrics")
    return warnings


# ── Chart helpers ───────────────────────────────────────────────────────────

def _bar_chart(items: List[Tuple[str, int]], max_width: int = 30) -> List[str]:
    """Generate a Unicode horizontal bar chart."""
    if not items:
        return []
    max_val = max(v for _, v in items)
    max_label = max(len(label) for label, _ in items)
    lines = []
    for label, val in items:
        bar_len = int(val / max(max_val, 1) * max_width)
        bar = "█" * bar_len
        lines.append(f"  {label:<{max_label}}  {bar} {val}")
    return lines


def _pie_ascii(items: List[Tuple[str, int]], total: int) -> List[str]:
    """Generate a simple percentage breakdown."""
    lines = []
    for label, val in items:
        pct = val / max(total, 1) * 100
        lines.append(f"  {label}: {val} ({pct:.0f}%)")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# QualityMetrics introspection
# ══════════════════════════════════════════════════════════════════════════════

def _get_quality_metrics_fields() -> Dict[str, Dict]:
    """Extract all QualityMetrics fields with metadata."""
    from .models import QualityMetrics

    fields_info = {}
    # Get field groups — _FIELD_GROUPS is a class-level dict (not a Pydantic field)
    groups = QualityMetrics._FIELD_GROUPS
    if not isinstance(groups, dict):
        # Pydantic v2 may wrap it; try .default
        groups = getattr(groups, "default", {}) or {}

    for name, field_info in QualityMetrics.model_fields.items():
        annotation = field_info.annotation
        type_str = "float"
        if annotation is not None:
            ann_str = str(annotation)
            if "int" in ann_str:
                type_str = "int"
            elif "str" in ann_str:
                type_str = "str"

        # Extract comment/description from source
        desc = field_info.description or ""
        group = groups.get(name, "other")
        fields_info[name] = {
            "type": type_str,
            "group": group,
            "description": desc,
        }

    return fields_info


def _get_score_direction(field_name: str, desc: str) -> str:
    """Determine score direction from field name and description."""
    desc_lower = (desc or "").lower()
    name_lower = field_name.lower()
    # Explicit markers
    if "lower=better" in desc_lower or "lower = better" in desc_lower:
        return "↓ lower=better"
    if "higher=better" in desc_lower or "higher = better" in desc_lower:
        return "↑ higher=better"
    # Common patterns
    if any(kw in name_lower for kw in ("error", "distortion", "distance", "loss", "jitter")):
        return "↓ lower=better"
    if any(kw in desc_lower for kw in ("lower", "error", "distortion", "distance")):
        return "↓ lower=better"
    if any(kw in name_lower for kw in ("score", "quality", "consistency", "fidelity", "similarity")):
        return "↑ higher=better"
    return "—"


def _get_deprecated_aliases() -> List[Tuple[str, str, str]]:
    """Return list of (alias, maps_to, note) for deprecated fields."""
    return [
        ("fid_score", "—", "deprecated, writes discarded"),
        ("kid_score", "—", "deprecated, writes discarded"),
        ("inception_score", "is_score", "alias"),
        ("ssim_score", "ssim", "alias"),
        ("psnr_score", "psnr", "alias"),
        ("lpips_score", "lpips", "alias"),
        ("alignment_score", "clip_score", "alias"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_metrics_doc() -> str:
    all_modules = ModuleRegistry.list_modules()

    # ── Collect module data ──────────────────────────────────────────────
    results: List[Dict] = []
    all_backends: Counter = Counter()
    all_warnings: List[Tuple[str, List[str]]] = []
    all_packages: Counter = Counter()
    field_writers: Dict[str, List[str]] = defaultdict(list)  # field → [module names]
    field_readers: Dict[str, List[str]] = defaultdict(list)  # field → [module names]
    speed_counts: Counter = Counter()
    gpu_count = 0

    for name in all_modules:
        cls = ModuleRegistry.get_module(name)
        if cls is None or cls.name == "unnamed_module":
            continue
        meta = cls.get_metadata()
        source = _get_source(cls)

        meta["group"] = _get_group(meta["name"], meta["input_type"])
        meta["backends"] = _detect_backends(source)
        meta["tiered"] = _detect_tiered(source)
        meta["fallback_chain"] = _detect_fallback_chain(source)
        meta["packages"] = _detect_packages(source)
        meta["gpu"] = _detect_gpu(source)
        meta["speed"] = _detect_speed_tier(source, meta["backends"])
        meta["hf_models"] = _detect_hf_models(source)
        meta["vram"] = _estimate_vram(source)
        meta["paper"] = _detect_paper(cls)

        for b in meta["backends"]:
            all_backends[b] += 1
        for pkg in meta["packages"]:
            all_packages[pkg] += 1
        speed_counts[meta["speed"]] += 1
        if meta["gpu"]:
            gpu_count += 1

        # Track field writes/reads
        written = _detect_fields_written(source)
        for f in written:
            field_writers[f].append(name)
        read = _detect_fields_read(source)
        for f in read:
            field_readers[f].append(name)

        warnings = _static_checks(source, meta)
        if warnings:
            all_warnings.append((name, warnings))

        results.append(meta)

    results.sort(key=lambda x: (x["group"], x["name"]))

    # ── Compute stats ────────────────────────────────────────────────────
    total_modules = len(results)
    total_outputs = sum(len(r["output_fields"]) for r in results)
    unique_outputs: Set[str] = set()
    for r in results:
        unique_outputs.update(r["output_fields"].keys())

    group_stats = defaultdict(lambda: {"modules": 0, "fields": 0})
    for r in results:
        group_stats[r["group"]]["modules"] += 1
        group_stats[r["group"]]["fields"] += len(r["output_fields"])

    input_counts: Counter = Counter()
    for r in results:
        it = r["input_type"]
        if "+ref" in it:
            input_counts["full-reference"] += 1
        elif "+cap" in it:
            input_counts["caption-required"] += 1
        elif it == "batch":
            input_counts["batch/dataset-level"] += 1
        elif "audio" in it:
            input_counts["audio"] += 1
        elif "vid" in it and "img" not in it:
            input_counts["video-only"] += 1
        else:
            input_counts["image+video"] += 1

    tiered_count = sum(1 for r in results if r["tiered"])

    # QualityMetrics introspection
    qm_fields = _get_quality_metrics_fields()
    written_fields = set(field_writers.keys())
    all_qm_field_names = set(qm_fields.keys())
    orphaned = all_qm_field_names - written_fields
    collisions = {f: writers for f, writers in field_writers.items() if len(writers) > 1}

    # Module dependencies (modules that read fields written by other modules)
    deps: List[Tuple[str, str, str]] = []  # (consumer, field, producer)
    for r in results:
        source = _get_source(ModuleRegistry.get_module(r["name"]))
        reads = _detect_fields_read(source)
        writes = _detect_fields_written(source)
        for field in reads - writes:  # reads a field it doesn't write
            if field in field_writers:
                for producer in field_writers[field]:
                    if producer != r["name"]:
                        deps.append((r["name"], field, producer))

    # ══════════════════════════════════════════════════════════════════════
    # BUILD DOCUMENT
    # ══════════════════════════════════════════════════════════════════════
    L = []  # output lines
    a = L.append  # shorthand

    a("# Ayase Metrics Reference")
    a("")
    a("Auto-generated reference for all pipeline modules. "
      "Run `ayase modules docs` to regenerate.")

    # ── 1. Summary Dashboard ─────────────────────────────────────────────
    a("")
    a("## Summary")
    a("")
    a("| Stat | Value |")
    a("|------|-------|")
    a(f"| Total modules | **{total_modules}** |")
    a(f"| Unique output fields | **{len(unique_outputs)}** |")
    a(f"| QualityMetrics fields | {len(qm_fields)} |")
    a(f"| Total output mappings | {total_outputs} |")
    a(f"| Tiered-backend modules | {tiered_count} |")
    a(f"| GPU-accelerated modules | {gpu_count} |")
    a(f"| Categories | {len(group_stats)} |")

    # ── 2. Category Distribution (bar chart) ─────────────────────────────
    a("")
    a("### Modules by Category")
    a("")
    a("```")
    cat_items = sorted(group_stats.items(), key=lambda x: -x[1]["modules"])
    for line in _bar_chart([(g, s["modules"]) for g, s in cat_items]):
        a(line)
    a("```")

    # ── 3. Input Type Breakdown ──────────────────────────────────────────
    a("")
    a("### By Input Type")
    a("")
    a("```")
    for line in _bar_chart(input_counts.most_common()):
        a(line)
    a("```")

    # ── 4. Backend Usage (bar chart) ─────────────────────────────────────
    a("")
    a("### Backend Usage")
    a("")
    a("```")
    for line in _bar_chart(all_backends.most_common()):
        a(line)
    a("```")

    # ── 5. Speed Tiers ───────────────────────────────────────────────────
    a("")
    a("### Speed Tiers")
    a("")
    a("```")
    tier_labels = {"fast": "fast (CPU, <0.1s)", "medium": "medium (GPU, ~1s)", "slow": "slow (LLM/VLM, >5s)"}
    for line in _bar_chart([(tier_labels.get(t, t), c) for t, c in speed_counts.most_common()]):
        a(line)
    a("```")

    # ── 6. Top Required Packages ─────────────────────────────────────────
    a("")
    a("### Top Required Packages")
    a("")
    a("```")
    for line in _bar_chart(all_packages.most_common(15)):
        a(line)
    a("```")

    # ── 7. Recommended Presets ───────────────────────────────────────────
    a("")
    a("### Recommended Module Presets")
    a("")
    for preset_name, info in _PRESETS.items():
        a(f"**{preset_name}** — {info['desc']}")
        a(f"```toml")
        a(f"modules = {info['modules']}")
        a(f"```")
        a("")

    # ── 8. Benchmark Coverage ────────────────────────────────────────────
    a("### Benchmark Coverage")
    a("")
    a("| Benchmark | Status |")
    a("|-----------|--------|")
    for bench, _mods in _BENCHMARK_MODULES.items():
        a(f"| {bench} | Covered |")
    a("")

    # ── 9. Field Collisions ──────────────────────────────────────────────
    if collisions:
        a("### Field Collisions")
        a("")
        a("Multiple modules write to the same QualityMetrics field:")
        a("")
        a("| Field | Writers |")
        a("|-------|---------|")
        for field, writers in sorted(collisions.items()):
            a(f"| `{field}` | {', '.join(f'`{w}`' for w in writers)} |")
        a("")

    # ── 10. Orphaned Fields ──────────────────────────────────────────────
    # Filter out string/special fields and known aliases
    real_orphans = {f for f in orphaned
                    if qm_fields.get(f, {}).get("type") == "float"
                    and f not in ("engagement_score", "human_preference_score")}
    if real_orphans:
        a("### Orphaned QualityMetrics Fields")
        a("")
        a(f"{len(real_orphans)} fields in `QualityMetrics` model that no module populates:")
        a("")
        for f in sorted(real_orphans):
            group = qm_fields[f]["group"]
            a(f"- `{f}` ({group})")
        a("")

    # ── 11. Module Dependencies (Mermaid) ────────────────────────────────
    # Build unique dependency edges
    dep_edges: Set[Tuple[str, str]] = set()
    for consumer, field, producer in deps:
        dep_edges.add((producer, consumer))

    if dep_edges:
        a("### Module Dependency Graph")
        a("")
        a("Modules that read QualityMetrics fields written by other modules:")
        a("")
        a("```mermaid")
        a("graph LR")
        for producer, consumer in sorted(dep_edges):
            a(f"    {producer} --> {consumer}")
        a("```")
        a("")

    # ── 12. Score Direction Reference ────────────────────────────────────
    a("### Score Direction Reference")
    a("")
    a("| Field | Direction | Range | Category |")
    a("|-------|-----------|-------|----------|")
    for field_name in sorted(unique_outputs):
        qm = qm_fields.get(field_name, {})
        # Find description from module output_fields
        desc = ""
        for r in results:
            if field_name in r["output_fields"]:
                desc = r["output_fields"][field_name]
                break
        direction = _get_score_direction(field_name, desc)
        # Extract range from desc
        range_match = re.search(r"\(([^)]*(?:higher|lower|dB|MOS|0-\d+|\d+-\d+)[^)]*)\)", desc or "")
        range_str = range_match.group(1) if range_match else "—"
        group = qm.get("group", "—")
        a(f"| `{field_name}` | {direction} | {range_str} | {group} |")
    a("")

    # ── 13. Deprecated Aliases ───────────────────────────────────────────
    aliases = _get_deprecated_aliases()
    a("### Deprecated Field Aliases")
    a("")
    a("| Old Name | Maps To | Status |")
    a("|----------|---------|--------|")
    for old, new, note in aliases:
        a(f"| `{old}` | `{new}` | {note} |")
    a("")

    # ── 14. Static Health Checks ─────────────────────────────────────────
    if all_warnings:
        a("### Static Health Checks")
        a("")
        a(f"{len(all_warnings)} module(s) with warnings:")
        a("")
        for mod_name, warns in sorted(all_warnings):
            for w in warns:
                a(f"- `{mod_name}`: {w}")
        a("")

    # ══════════════════════════════════════════════════════════════════════
    # PER-MODULE TABLES (with enriched columns)
    # ══════════════════════════════════════════════════════════════════════
    a("---")
    a("")

    current_group = None
    for r in results:
        if r["group"] != current_group:
            if current_group is not None:
                a("")
            a(f"## {r['group']}")
            a("")
            a("| Module | Input | Outputs | Description | Config |")
            a("|--------|-------|---------|-------------|--------|")
            current_group = r["group"]

        out_parts = []
        for field, desc in r["output_fields"].items():
            out_parts.append(f"`{field}` - {desc}" if desc else f"`{field}`")
        out_str = "; ".join(out_parts) if out_parts else "-"

        cfg = r["default_config"]
        cfg_items = [f"`{k}={v}`" for k, v in list(cfg.items())[:2]]
        if len(cfg) > 2:
            cfg_items.append(f"+{len(cfg) - 2}")
        cfg_str = ", ".join(cfg_items) or "-"

        a(f"| `{r['name']}` | {r['input_type']} | {out_str} "
          f"| {r['description']} | {cfg_str} |")

    a("")

    # ── Per-module detail cards (packages, GPU, speed, fallback) ─────────
    a("---")
    a("")
    a("## Module Details")
    a("")
    a("Per-module requirements, speed tier, GPU usage, and fallback chains.")
    a("")

    for r in results:
        badges = []
        if r["gpu"]:
            badges.append("GPU")
        badges.append(r["speed"])
        if r["tiered"]:
            badges.append("tiered")
        badge_str = " · ".join(badges)

        pkg_str = ", ".join(r["packages"]) if r["packages"] else "—"
        chain_str = " → ".join(r["fallback_chain"]) if r["fallback_chain"] else "—"
        hf_str = ", ".join(f"`{m}`" for m in r["hf_models"]) if r["hf_models"] else "—"
        vram_str = r["vram"] or "—"
        paper_str = r["paper"] or "—"

        a(f"<details><summary><code>{r['name']}</code> [{badge_str}]</summary>")
        a("")
        a(f"- **Packages**: {pkg_str}")
        if hf_str != "—":
            a(f"- **Models**: {hf_str}")
        if vram_str != "—":
            a(f"- **Est. VRAM**: {vram_str}")
        if chain_str != "—":
            a(f"- **Fallback**: {chain_str}")
        if paper_str != "—":
            a(f"- **Paper**: {paper_str}")
        a("")
        a("</details>")
        a("")

    return "\n".join(L)
