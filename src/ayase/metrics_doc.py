"""Generate METRICS.md from module metadata via PipelineModule.get_metadata().

Produces metric-centric tables grouped by QualityMetrics category, where each
row is a (metric, module) pair with direction, range, speed, GPU, backend,
source links, and description — all in one place.

Integrity issues (field collisions, orphans, static checks) are reported as
stderr warnings during generation, not included in the output.
"""

import inspect
import re
from collections import Counter, defaultdict
from pathlib import Path
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
    # Pattern 1: sample.quality_metrics.FIELD =
    for m in re.finditer(r"quality_metrics\.(\w+)\s*=", source):
        writes.add(m.group(1))
    # Pattern 2: metric_field = "FIELD" (ReferenceBasedModule/NoReferenceModule auto-assignment)
    for m in re.finditer(r'metric_field\s*=\s*["\'](\w+)["\']', source):
        writes.add(m.group(1))
    # Pattern 3: qm.FIELD = (variable alias for quality_metrics)
    if re.search(r"\bqm\s*=\s*\w*\.?quality_metrics", source):
        for m in re.finditer(r"\bqm\.(\w+)\s*=", source):
            writes.add(m.group(1))
    return writes


_UTILITY_MODULES = {"embedding", "dedup", "diversity_selection", "knowledge_graph"}

# ── Metric category display names and ordering ────────────────────────────
_CATEGORY_DISPLAY = {
    "nr_quality": "No-Reference Quality",
    "fr_quality": "Full-Reference Quality",
    "alignment": "Text-Video Alignment",
    "temporal": "Temporal Consistency",
    "motion": "Motion & Dynamics",
    "basic": "Basic Visual Quality",
    "aesthetic": "Aesthetics",
    "audio": "Audio Quality",
    "face": "Face & Identity",
    "scene": "Scene & Content",
    "hdr": "HDR & Color",
    "codec": "Codec & Technical",
    "spatial": "Depth & Spatial",
    "production": "Production Quality",
    "text": "OCR & Text",
    "safety": "Safety & Ethics",
    "i2v": "Image-to-Video Reference",
    "meta": "Meta & Curation",
}
_CATEGORY_ORDER = list(_CATEGORY_DISPLAY.keys())


def _detect_source_links(source: str, cls: type) -> str:
    """Extract source links (paper, GitHub, HF) as compact markdown."""
    links = []
    # arXiv
    for m in re.finditer(r"(https?://arxiv\.org/abs/[\w.]+)", source):
        links.append(f"[arXiv]({m.group(1)})")
        break
    # GitHub
    for m in re.finditer(r"(https?://github\.com/[\w-]+/[\w.-]+)", source):
        links.append(f"[GitHub]({m.group(1)})")
        break
    # HuggingFace
    for m in re.finditer(r'["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)["\']', source):
        candidate = m.group(1)
        if not any(x in candidate for x in ("http", "path", "file", "dir", "main/")):
            links.append(f"[HF](https://huggingface.co/{candidate})")
            break
    # Paper citation from docstring
    if not links:
        doc = cls.__doc__ or ""
        for m_p in re.finditer(
            r"((?:CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|WACV|TIP|TPAMI)\s*\d{4})", doc
        ):
            links.append(m_p.group(1))
            break
    return " · ".join(links) if links else "—"


def _static_checks(source: str, meta: Dict, cls: type = None) -> List[str]:
    warnings = []
    # For compat alias subclasses, include parent source in checks
    full_source = source
    if cls is not None and len(source) < 300:
        for base in cls.__mro__[1:]:
            if base.__module__.startswith("ayase."):
                try:
                    full_source = source + "\n" + inspect.getsource(base)
                    break
                except (TypeError, OSError):
                    pass

    # Skip checks for known utility modules (no quality_metrics by design)
    if meta.get("name") in _UTILITY_MODULES:
        return warnings

    if "def on_mount(" in full_source and "def setup(" not in full_source:
        if "super().on_mount()" not in full_source:
            warnings.append("uses `on_mount()` instead of `setup()`")
    # A module "produces output" if it has output_fields, validation_issues,
    # metric_field, add_dataset_metric, or is a batch/dataset-level module.
    has_output = (
        meta["output_fields"]
        or "validation_issues" in full_source
        or re.search(r'metric_field\s*=\s*["\']', full_source)
        or "add_dataset_metric" in full_source
        or "BatchMetricModule" in full_source
    )
    if not has_output:
        warnings.append("no output fields and no validation issues")
    # Check that modules declaring output_fields actually write them
    writes_metrics = (
        "quality_metrics" in full_source
        or re.search(r'metric_field\s*=\s*["\']', full_source)
        or re.search(r"\bqm\s*=\s*\w*\.?quality_metrics", full_source)
    )
    if meta["output_fields"] and not writes_metrics:
        warnings.append("declares output fields but never assigns quality_metrics")
    return warnings


# ── Chart helpers ───────────────────────────────────────────────────────────

_CHART_WIDTH = 900
_CHART_HEIGHT_BAR = 400
_CHART_SCALE = 2  # retina-quality PNG export


def _generate_charts(
    cat_items: List[Tuple[str, Dict]],
    input_counts: "Counter",
    speed_counts: "Counter",
    all_backends: "Counter",
    output_dir: Path,
    all_packages: Optional["Counter"] = None,
    summary_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """Generate PNG chart images using Plotly.

    All charts are rendered at uniform width (_CHART_WIDTH px).
    Returns dict of chart_name -> relative path to PNG file.
    """
    paths: Dict[str, str] = {}
    try:
        import plotly.graph_objects as go

        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Shared layout defaults ─────────────────────────────────────
        _layout = dict(
            font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # ── 1. Category horizontal bar (cleaner than pie for 13+ categories)
        labels = [g for g, _ in cat_items]
        values = [s["modules"] for _, s in cat_items]
        colors = [
            "#4DC9F6", "#F67019", "#F53794", "#537BC4", "#ACC236",
            "#166A8F", "#00A950", "#58595B", "#8549BA", "#FFC300",
            "#C70039", "#DAF7A6", "#FF5733",
        ][:len(labels)]
        fig = go.Figure(go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h",
            marker_color=colors[::-1],
            text=values[::-1], textposition="outside", textfont_size=13,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH,
            height=max(400, len(labels) * 38 + 80),
            title=dict(text="Module Distribution by Category", font_size=16, x=0.5),
            xaxis=dict(title="Number of Modules", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),
            bargap=0.25,
            showlegend=False,
        )
        p = output_dir / "chart_categories.png"
        fig.write_image(str(p), scale=_CHART_SCALE)
        paths["categories"] = f"docs/{p.name}"

        # ── 2. Input type donut ────────────────────────────────────────
        i_labels = [l for l, _ in input_counts.most_common()]
        i_values = [c for _, c in input_counts.most_common()]
        fig = go.Figure(go.Pie(
            labels=i_labels, values=i_values,
            hole=0.5, textinfo="label+percent",
            textposition="outside", textfont_size=11,
            marker=dict(
                colors=["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"],
                line=dict(color="white", width=2),
            ),
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=450,
            title=dict(text="Input Type Distribution", font_size=16, x=0.5),
            legend=dict(font_size=11),
        )
        p = output_dir / "chart_input_types.png"
        fig.write_image(str(p), scale=_CHART_SCALE)
        paths["input_types"] = f"docs/{p.name}"

        # ── 3. Speed tiers bar ─────────────────────────────────────────
        tier_map = {"fast": "Fast (CPU)", "medium": "Medium (GPU)", "slow": "Slow (LLM/VLM)"}
        tier_colors = {"fast": "#2ECC71", "medium": "#F39C12", "slow": "#E74C3C"}
        s_items = speed_counts.most_common()
        s_labels = [tier_map.get(t, t) for t, _ in s_items]
        s_values = [c for _, c in s_items]
        s_cols = [tier_colors.get(t, "#3498DB") for t, _ in s_items]
        fig = go.Figure(go.Bar(
            x=s_values, y=s_labels, orientation="h",
            marker_color=s_cols,
            text=s_values, textposition="outside", textfont_size=14,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=_CHART_HEIGHT_BAR,
            title=dict(text="Speed Tiers", font_size=16, x=0.5),
            xaxis=dict(title="Number of Modules", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),
            bargap=0.35,
        )
        p = output_dir / "chart_speed.png"
        fig.write_image(str(p), scale=_CHART_SCALE)
        paths["speed"] = f"docs/{p.name}"

        # ── 4. Backend usage bar ───────────────────────────────────────
        b_items = all_backends.most_common(10)
        b_labels = [l for l, _ in b_items]
        b_values = [c for _, c in b_items]
        fig = go.Figure(go.Bar(
            x=b_values, y=b_labels, orientation="h",
            marker_color="#5DADE2",
            text=b_values, textposition="outside", textfont_size=13,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=max(_CHART_HEIGHT_BAR, len(b_labels) * 42 + 80),
            title=dict(text="Backend Usage (Top 10)", font_size=16, x=0.5),
            xaxis=dict(title="Number of Modules", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(autorange="reversed"),
            bargap=0.3,
        )
        p = output_dir / "chart_backends.png"
        fig.write_image(str(p), scale=_CHART_SCALE)
        paths["backends"] = f"docs/{p.name}"

        # ── 5. Top packages bar ────────────────────────────────────────
        if all_packages:
            pk_items = all_packages.most_common(12)
            pk_labels = [l for l, _ in pk_items]
            pk_values = [c for _, c in pk_items]
            fig = go.Figure(go.Bar(
                x=pk_values, y=pk_labels, orientation="h",
                marker_color="#8E44AD",
                text=pk_values, textposition="outside", textfont_size=13,
            ))
            fig.update_layout(
                **_layout, width=_CHART_WIDTH,
                height=max(_CHART_HEIGHT_BAR, len(pk_labels) * 38 + 80),
                title=dict(text="Top Required Packages", font_size=16, x=0.5),
                xaxis=dict(title="Number of Modules", showgrid=True, gridcolor="#f0f0f0"),
                yaxis=dict(autorange="reversed"),
                bargap=0.3,
            )
            p = output_dir / "chart_packages.png"
            fig.write_image(str(p), scale=_CHART_SCALE)
            paths["packages"] = f"docs/{p.name}"

        # ── 6. Summary dashboard (number indicators) ───────────────────
        if summary_stats:
            fig = go.Figure()
            keys = list(summary_stats.keys())
            n = len(keys)
            for i, (label, value) in enumerate(summary_stats.items()):
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=value,
                    title={"text": label, "font": {"size": 14}},
                    number={"font": {"size": 36, "color": "#2C3E50"}},
                    domain={"x": [i / n + 0.01, (i + 1) / n - 0.01], "y": [0.1, 0.9]},
                ))
            fig.update_layout(
                width=_CHART_WIDTH, height=160,
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            p = output_dir / "chart_summary.png"
            fig.write_image(str(p), scale=_CHART_SCALE)
            paths["summary"] = f"docs/{p.name}"

    except ImportError:
        pass  # plotly/kaleido not available, skip chart generation
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Chart generation failed: {exc}")

    return paths


def _collect_test_status(run_tests: bool = True) -> Dict[str, Dict[str, bool]]:
    """Collect test pass/fail status for modules by running pytest in light mode.

    Args:
        run_tests: If True, actually run pytest. If False, return empty dict.

    Returns dict: module_name -> {"light": True/False}
    """
    if not run_tests:
        return {}

    import subprocess
    result: Dict[str, Dict[str, bool]] = {}
    project_root = str(Path(__file__).parent.parent.parent)

    try:
        # Run per-module basics tests (ultra-fast, ~1s for all 312)
        import os as _os
        env = {**_os.environ, "AYASE_TEST_MODE": "1"}
        proc = subprocess.run(
            ["python", "-m", "pytest", "tests/modules/per_module/",
             "-k", "basics", "--tb=no", "--no-header", "-v"],
            capture_output=True, text=True, timeout=30,
            cwd=project_root, encoding="utf-8", errors="replace",
            env=env,
        )
        for line in proc.stdout.splitlines():
            if "PASSED" not in line and "FAILED" not in line:
                continue
            # Format: tests/modules/test_x.py::test_name PASSED  [ 10%]
            passed = " PASSED" in line

            # Extract test function name from path::test_name
            if "::" not in line:
                continue
            test_func = line.split("::")[1].split()[0]  # get "test_name" before PASSED/FAILED

            # Strip test_ prefix
            if not test_func.startswith("test_"):
                continue
            name = test_func[5:]  # remove "test_"

            # Strip known suffixes to get module name
            suffixes = [
                "_basics", "_image", "_video", "_no_reference", "_with_reference",
                "_no_metadata", "_with_metadata", "_image_skipped",
                "_is_batch", "_extract", "_fields_exist", "_fields",
                "_no_pointcloud", "_config",
            ]
            mod_name = name
            for suffix in sorted(suffixes, key=len, reverse=True):
                if name.endswith(suffix):
                    mod_name = name[: -len(suffix)]
                    break

            if not mod_name:
                continue

            if mod_name not in result:
                result[mod_name] = {"light": True}
            if not passed:
                result[mod_name]["light"] = False

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as exc:
        import logging
        logging.getLogger(__name__).debug(f"Test collection failed: {exc}")

    return result


def _format_test_status(module_name: str, test_results: Dict) -> str:
    """Format test status as emoji checkmarks.

    Icons: ✅ = pass, ❌ = fail, ⏳ = not run yet
    Two columns: light (heuristic) | full (ML models)
    """
    if module_name not in test_results:
        return "\u2014"
    info = test_results[module_name]
    light = "\u2705" if info.get("light") else "\u274c"
    full = "\u2705" if info.get("full") else "\u23f3"  # hourglass = not run
    return f"{light}{full}"


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



# ══════════════════════════════════════════════════════════════════════════════
# Main generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_metrics_doc(run_tests: bool = True) -> str:
    """Generate METRICS.md content with charts, test status, and version header.

    Args:
        run_tests: If True, run pytest to collect test pass/fail status
                   for each module. Adds checkmark emojis to module tables.
    """
    ModuleRegistry.discover_modules()
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

        warnings = _static_checks(source, meta, cls)
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

    # ── Run tests to get status (optional) ────────────────────────────────
    test_results = _collect_test_status(run_tests=run_tests)

    # ── Generate charts ──────────────────────────────────────────────────
    cat_items = sorted(group_stats.items(), key=lambda x: -x[1]["modules"])
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    summary_stats = {
        "Modules": total_modules,
        "Output Fields": len(unique_outputs),
        "QM Fields": len(qm_fields),
        "Tiered": tiered_count,
        "GPU": gpu_count,
        "Categories": len(group_stats),
    }
    chart_paths = _generate_charts(
        cat_items, input_counts, speed_counts, all_backends, docs_dir,
        all_packages=all_packages, summary_stats=summary_stats,
    )

    # ══════════════════════════════════════════════════════════════════════
    # BUILD DOCUMENT
    # ══════════════════════════════════════════════════════════════════════
    L = []  # output lines
    a = L.append  # shorthand

    # ── Header with version + date ────────────────────────────────────────
    from datetime import datetime
    try:
        from ayase import __version__
    except ImportError:
        __version__ = "dev"
    a("# Ayase Metrics Reference")
    a("")
    a(f"> **Version {__version__}** · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
      f"· **{total_modules} modules** · **{len(qm_fields)} metrics**")
    a(">")
    a("> `ayase modules docs -o METRICS.md` to regenerate")
    a(">")
    a(f"> Tests: `pytest tests/` (light) · `pytest tests/ --full` (with ML models)")

    # ── 1. Summary Dashboard ─────────────────────────────────────────────
    a("")
    a("## Summary")
    a("")
    if "summary" in chart_paths:
        a(f"![Summary Dashboard]({chart_paths['summary']})")
    else:
        a(f"**{total_modules}** modules · **{len(unique_outputs)}** output fields "
          f"· **{len(qm_fields)}** QualityMetrics fields · **{tiered_count}** tiered "
          f"· **{gpu_count}** GPU · **{len(group_stats)}** categories")

    # ── 2. Category Distribution ──────────────────────────────────────────
    a("")
    a("### Modules by Category")
    a("")
    if "categories" in chart_paths:
        a(f"![Module Distribution by Category]({chart_paths['categories']})")
    else:
        a("```")
        for line in _bar_chart([(g, s["modules"]) for g, s in cat_items]):
            a(line)
        a("```")

    # ── 3. Input Type Breakdown ──────────────────────────────────────────
    a("")
    a("### By Input Type")
    a("")
    if "input_types" in chart_paths:
        a(f"![Input Type Distribution]({chart_paths['input_types']})")
    else:
        a("```")
        for line in _bar_chart(input_counts.most_common()):
            a(line)
        a("```")

    # ── 4. Speed Tiers ───────────────────────────────────────────────────
    a("")
    a("### Speed Tiers")
    a("")
    if "speed" in chart_paths:
        a(f"![Speed Tiers]({chart_paths['speed']})")
    else:
        a("```")
        tier_labels = {"fast": "fast (CPU, <0.1s)", "medium": "medium (GPU, ~1s)", "slow": "slow (LLM/VLM, >5s)"}
        for line in _bar_chart([(tier_labels.get(t, t), c) for t, c in speed_counts.most_common()]):
            a(line)
        a("```")

    # ── 5. Backend Usage ──────────────────────────────────────────────────
    a("")
    a("### Backend Usage")
    a("")
    if "backends" in chart_paths:
        a(f"![Backend Usage]({chart_paths['backends']})")
    else:
        a("```")
        for line in _bar_chart(all_backends.most_common()):
            a(line)
        a("```")

    # ── 6. Top Required Packages ─────────────────────────────────────────
    a("")
    a("### Top Required Packages")
    a("")
    if "packages" in chart_paths:
        a(f"![Top Required Packages]({chart_paths['packages']})")
    else:
        a("```")
        for line in _bar_chart(all_packages.most_common(15)):
            a(line)
        a("```")


    # ── Integrity warnings (stderr only, not in output) ────────────────
    import sys

    # Filter out guarded collisions (writer checks `if field is None` before writing)
    real_collisions = {}
    for field, writers in collisions.items():
        unguarded = []
        for w in writers:
            w_cls = ModuleRegistry.get_module(w)
            if w_cls is None:
                unguarded.append(w)
                continue
            w_src = _get_source(w_cls)
            if f"{field} is None" not in w_src:
                unguarded.append(w)
        if len(unguarded) > 1:
            real_collisions[field] = unguarded
    if real_collisions:
        print(f"WARNING: {len(real_collisions)} field collision(s):", file=sys.stderr)
        for field, writers in sorted(real_collisions.items()):
            print(f"  {field}: {', '.join(writers)}", file=sys.stderr)
    real_orphans = {f for f in orphaned
                    if qm_fields.get(f, {}).get("type") == "float"
                    and f not in ("engagement_score", "human_preference_score")}
    if real_orphans:
        print(f"WARNING: {len(real_orphans)} orphaned QualityMetrics field(s):", file=sys.stderr)
        for f in sorted(real_orphans):
            print(f"  {f}", file=sys.stderr)

    if all_warnings:
        print(f"WARNING: {len(all_warnings)} module(s) with static health issues:", file=sys.stderr)
        for mod_name, warns in sorted(all_warnings):
            for w in warns:
                print(f"  {mod_name}: {w}", file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════════
    # METRIC-CENTRIC TABLES (grouped by _FIELD_GROUPS category)
    # ══════════════════════════════════════════════════════════════════════

    # Build module lookup: name → enriched result dict
    mod_lookup: Dict[str, Dict] = {r["name"]: r for r in results}

    # Build source links cache (per module, computed once)
    source_links_cache: Dict[str, str] = {}
    for name in all_modules:
        cls = ModuleRegistry.get_module(name)
        if cls is None:
            continue
        source_links_cache[name] = _detect_source_links(_get_source(cls), cls)

    # Build metric rows: one per (field, writer_module) pair
    speed_badges = {"fast": "\u26a1", "medium": "\u23f1\ufe0f", "slow": "\U0001f40c"}
    metric_rows: Dict[str, list] = defaultdict(list)  # category → rows

    for field_name in sorted(qm_fields.keys()):
        qm = qm_fields[field_name]
        group = qm["group"]
        writers = field_writers.get(field_name, [])
        if not writers:
            continue

        # Get description and direction from QualityMetrics field comment
        desc = ""
        for r in results:
            if field_name in r["output_fields"]:
                desc = r["output_fields"][field_name]
                break
        direction = _get_score_direction(field_name, desc)
        range_match = re.search(
            r"\(([^)]*(?:higher|lower|dB|MOS|0-\d+|\d+-\d+)[^)]*)\)", desc or ""
        )
        range_str = range_match.group(1) if range_match else "—"

        for mod_name in sorted(writers):
            mod = mod_lookup.get(mod_name)
            if mod is None:
                continue
            badge = speed_badges.get(mod["speed"], "")
            chain = " → ".join(mod["fallback_chain"]) if mod["fallback_chain"] else "—"
            gpu = "✓" if mod["gpu"] else ""
            source = source_links_cache.get(mod_name, "—")
            test_status = _format_test_status(mod_name, test_results)

            metric_rows[group].append(
                f"| `{field_name}` | {direction} | {range_str} "
                f"| `{mod_name}` | {mod['input_type']} "
                f"| {badge} {mod['speed']} | {gpu} | {chain} "
                f"| {source} | {test_status} | {mod['description']} |"
            )

    a("---")
    a("")

    for cat_key in _CATEGORY_ORDER:
        rows = metric_rows.get(cat_key)
        if not rows:
            continue
        display = _CATEGORY_DISPLAY.get(cat_key, cat_key)
        a(f"## {display} ({len(rows)} metrics)")
        a("")
        a("| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |")
        a("|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|")
        for row in rows:
            a(row)
        a("")

    # Handle "other" category (fields not in _CATEGORY_ORDER)
    other_rows = metric_rows.get("other")
    if other_rows:
        a(f"## Other ({len(other_rows)} metrics)")
        a("")
        a("| Metric | Dir | Range | Module | Input | Speed | GPU | Backend | Source | Test | Description |")
        a("|--------|-----|-------|--------|-------|-------|-----|---------|--------|------|-------------|")
        for row in other_rows:
            a(row)
        a("")

    # ── Utility & Validation Modules (no metric output) ───────────────
    no_output = [r for r in results if not r["output_fields"]]
    if no_output:
        a(f"## Utility & Validation ({len(no_output)} modules)")
        a("")
        a("Modules that perform validation, embedding, deduplication, or dataset-level "
          "analysis without writing individual QualityMetrics fields.")
        a("")
        a("| Module | Input | Speed | GPU | Description |")
        a("|--------|-------|-------|-----|-------------|")
        for r in sorted(no_output, key=lambda x: x["name"]):
            badge = speed_badges.get(r["speed"], "")
            gpu = "✓" if r["gpu"] else ""
            a(f"| `{r['name']}` | {r['input_type']} | {badge} {r['speed']} | {gpu} | {r['description']} |")
        a("")

    return "\n".join(L)
