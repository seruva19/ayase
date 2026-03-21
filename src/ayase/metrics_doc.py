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


def _get_module_file_link(cls: type) -> str:
    """Get relative path to module source file for repo linking."""
    try:
        fpath = Path(inspect.getfile(cls))
        # Walk up to find src/ayase/modules/...
        parts = fpath.parts
        try:
            src_idx = parts.index("src")
            return "/".join(parts[src_idx:])
        except ValueError:
            return fpath.name
    except (TypeError, OSError):
        return ""


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

_CHART_WIDTH = 480
_CHART_HEIGHT_BAR = 280
_CHART_SCALE = 2  # retina-quality PNG export


def _generate_charts(
    cat_items: List[Tuple[str, Dict]],
    input_counts: "Counter",
    speed_counts: "Counter",
    all_backends: "Counter",
    output_dir: Path,
    all_packages: Optional["Counter"] = None,
    summary_stats: Optional[Dict[str, int]] = None,
    metrics_per_cat: Optional[Dict[str, int]] = None,
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
            font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=11),
            margin=dict(l=10, r=30, t=8, b=10),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#fafafa",
        )

        # All paired charts use the same height for alignment
        _PAIR_H = _CHART_HEIGHT_BAR

        # ── 1. Category horizontal bar ─────────────────────────────────
        labels = [g for g, _ in cat_items]
        values = [s["modules"] for _, s in cat_items]
        colors = [
            "#6C5CE7", "#00B894", "#FD79A8", "#0984E3", "#FDCB6E",
            "#E17055", "#00CEC9", "#636E72", "#A29BFE", "#FAB1A0",
            "#55EFC4", "#DFE6E9", "#74B9FF",
        ][:len(labels)]
        fig = go.Figure(go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h",
            marker=dict(color=colors[::-1], line=dict(width=0)),
            text=values[::-1], textposition="outside", textfont_size=9,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=_PAIR_H,
            title=None,
            xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
            bargap=0.12,
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
            hole=0.45, textinfo="label+percent",
            textposition="inside", textfont_size=9,
            insidetextorientation="horizontal",
            marker=dict(
                colors=["#74B9FF", "#FD79A8", "#FFEAA7", "#55EFC4", "#A29BFE", "#FAB1A0"],
                line=dict(color="#fafafa", width=2),
            ),
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=_PAIR_H,
            title=None,
            legend=dict(font_size=9, bgcolor="rgba(0,0,0,0)"),
        )
        p = output_dir / "chart_input_types.png"
        fig.write_image(str(p), scale=_CHART_SCALE)
        paths["input_types"] = f"docs/{p.name}"

        # ── 3. Speed tiers bar ─────────────────────────────────────────
        tier_map = {"fast": "Fast (CPU)", "medium": "Medium (GPU)", "slow": "Slow (LLM/VLM)"}
        tier_colors = {"fast": "#00B894", "medium": "#FDCB6E", "slow": "#E17055"}
        s_items = speed_counts.most_common()
        s_labels = [tier_map.get(t, t) for t, _ in s_items]
        s_values = [c for _, c in s_items]
        s_cols = [tier_colors.get(t, "#74B9FF") for t, _ in s_items]
        fig = go.Figure(go.Bar(
            x=s_values, y=s_labels, orientation="h",
            marker=dict(color=s_cols, line=dict(width=0)),
            text=s_values, textposition="outside", textfont_size=11,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=_PAIR_H,
            title=None,
            xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False),
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
            marker=dict(color="#74B9FF", line=dict(width=0)),
            text=b_values, textposition="outside", textfont_size=10,
        ))
        fig.update_layout(
            **_layout, width=_CHART_WIDTH, height=_PAIR_H,
            title=None,
            xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
            bargap=0.2,
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
                marker=dict(color="#A29BFE", line=dict(width=0)),
                text=pk_values, textposition="outside", textfont_size=9,
            ))
            fig.update_layout(
                **_layout, width=_CHART_WIDTH, height=_PAIR_H,
                title=None,
                xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                bargap=0.2,
            )
            p = output_dir / "chart_packages.png"
            fig.write_image(str(p), scale=_CHART_SCALE)
            paths["packages"] = f"docs/{p.name}"

        # ── 6. Metrics per category bar ────────────────────────────────
        if metrics_per_cat:
            mc_items = sorted(metrics_per_cat.items(), key=lambda x: x[1])
            mc_labels = [_CATEGORY_DISPLAY.get(k, k) for k, _ in mc_items]
            mc_values = [v for _, v in mc_items]
            fig = go.Figure(go.Bar(
                x=mc_values, y=mc_labels, orientation="h",
                marker=dict(color="#00B894", line=dict(width=0)),
                text=mc_values, textposition="outside", textfont_size=9,
            ))
            fig.update_layout(
                **_layout, width=_CHART_WIDTH, height=_PAIR_H,
                title=None,
                xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
                bargap=0.12,
            )
            p = output_dir / "chart_metrics_per_cat.png"
            fig.write_image(str(p), scale=_CHART_SCALE)
            paths["metrics_per_cat"] = f"docs/{p.name}"

        # ── 7. Summary dashboard (number indicators) ───────────────────
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
                width=_CHART_WIDTH * 2, height=120,
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
    """Extract all QualityMetrics fields with metadata and inline comments."""
    from .models import QualityMetrics

    fields_info = {}
    groups = QualityMetrics._FIELD_GROUPS
    if not isinstance(groups, dict):
        groups = getattr(groups, "default", {}) or {}

    # Extract inline comments from source (e.g. "blur_score: Optional[float] = None  # Laplacian variance")
    inline_comments: Dict[str, str] = {}
    try:
        src = inspect.getsource(QualityMetrics)
        for m in re.finditer(r"(\w+):\s*Optional\[.*?\]\s*=\s*None\s*#\s*(.*)", src):
            inline_comments[m.group(1)] = m.group(2).strip()
    except (TypeError, OSError):
        pass

    for name, field_info in QualityMetrics.model_fields.items():
        annotation = field_info.annotation
        type_str = "float"
        if annotation is not None:
            ann_str = str(annotation)
            if "int" in ann_str:
                type_str = "int"
            elif "str" in ann_str:
                type_str = "str"

        desc = field_info.description or ""
        group = groups.get(name, "other")
        comment = inline_comments.get(name, "")
        fields_info[name] = {
            "type": type_str,
            "group": group,
            "description": desc,
            "comment": comment,
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
    # Count metrics per _FIELD_GROUPS category
    metrics_per_cat_count: Dict[str, int] = defaultdict(int)
    for field_name, info in qm_fields.items():
        if field_writers.get(field_name):  # only count fields with writers
            metrics_per_cat_count[info["group"]] += 1

    chart_paths = _generate_charts(
        cat_items, input_counts, speed_counts, all_backends, docs_dir,
        all_packages=all_packages, summary_stats=summary_stats,
        metrics_per_cat=dict(metrics_per_cat_count),
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

    # ── Charts (two per row via HTML table) ─────────────────────────────
    chart_titles = {
        "categories": "Modules by Category",
        "input_types": "Input Types",
        "speed": "Speed Tiers",
        "backends": "Backend Usage",
        "packages": "Top Packages",
        "metrics_per_cat": "Metrics per Category",
    }
    chart_keys = [k for k in ("categories", "input_types", "speed", "backends", "packages", "metrics_per_cat") if k in chart_paths]
    if chart_keys:
        a("")
        for i in range(0, len(chart_keys), 2):
            pair = chart_keys[i:i+2]
            if len(pair) == 2:
                t1 = chart_titles.get(pair[0], "")
                t2 = chart_titles.get(pair[1], "")
                a("<table><tr>")
                p0 = chart_paths[pair[0]]
                p1 = chart_paths[pair[1]]
                a(f'<td valign="top"><h4>{t1}</h4><img src="{p0}"/></td>')
                a(f'<td valign="top"><h4>{t2}</h4><img src="{p1}"/></td>')
                a("</tr></table>")
            else:
                t = chart_titles.get(pair[0], "")
                a(f"<h4>{t}</h4>")
                a("")
                a(f"![]({chart_paths[pair[0]]})")
            a("")

    # ── Text fallbacks (when plotly is unavailable) ───────────────────
    if not chart_paths:
        a("")
        a("```")
        for line in _bar_chart([(g, s["modules"]) for g, s in cat_items]):
            a(line)
        a("```")
        a("")
        a("```")
        for line in _bar_chart(input_counts.most_common()):
            a(line)
        a("```")
        a("")
        a("```")
        tier_labels = {"fast": "fast (CPU, <0.1s)", "medium": "medium (GPU, ~1s)", "slow": "slow (LLM/VLM, >5s)"}
        for line in _bar_chart([(tier_labels.get(t, t), c) for t, c in speed_counts.most_common()]):
            a(line)
        a("```")
        a("")
        a("```")
        for line in _bar_chart(all_backends.most_common()):
            a(line)
        a("```")
        a("")
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
    # METRIC INFO PANELS (grouped by _FIELD_GROUPS category)
    # ══════════════════════════════════════════════════════════════════════

    mod_lookup: Dict[str, Dict] = {r["name"]: r for r in results}
    speed_badges = {"fast": "\u26a1", "medium": "\u23f1\ufe0f", "slow": "\U0001f40c"}

    # Pre-compute per-module: source links, file paths, packages
    mod_extra: Dict[str, Dict] = {}
    for name in all_modules:
        cls = ModuleRegistry.get_module(name)
        if cls is None:
            continue
        src = _get_source(cls)
        mod_extra[name] = {
            "source_links": _detect_source_links(src, cls),
            "file_path": _get_module_file_link(cls),
        }

    # Build field readers: which modules READ each field (dependencies)
    field_reader_names: Dict[str, List[str]] = defaultdict(list)
    for consumer, field, producer in deps:
        if consumer not in field_reader_names[field]:
            field_reader_names[field].append(consumer)

    # Group metrics by category
    metrics_by_cat: Dict[str, list] = defaultdict(list)
    for field_name in sorted(qm_fields.keys()):
        group = qm_fields[field_name]["group"]
        writers = field_writers.get(field_name, [])
        if writers:
            metrics_by_cat[group].append(field_name)

    # ── Category navigation ─────────────────────────────────────────────
    a("")
    nav_parts = []
    for cat_key in _CATEGORY_ORDER:
        fields = metrics_by_cat.get(cat_key)
        if not fields:
            continue
        display = _CATEGORY_DISPLAY.get(cat_key, cat_key)
        # GitHub heading anchor: lowercase, spaces→hyphens, strip non-alnum except hyphens
        anchor = re.sub(r"[^a-z0-9 -]", "", display.lower()).replace(" ", "-")
        anchor = f"{anchor}-{len(fields)}-metrics"
        nav_parts.append(f"[{display}](#{anchor}) ({len(fields)})")
    no_out_count = len([r for r in results if not r["output_fields"]])
    nav_parts.append(f"[Utility & Validation](#utility--validation-{no_out_count}-modules) ({no_out_count})")
    a(" · ".join(nav_parts))
    a("")
    a("---")
    a("")

    for cat_key in list(_CATEGORY_ORDER) + ["other"]:
        fields = metrics_by_cat.get(cat_key)
        if not fields:
            continue
        display = _CATEGORY_DISPLAY.get(cat_key, "Other")
        a(f"## {display} ({len(fields)} metrics)")
        a("")

        for field_name in fields:
            writers = field_writers.get(field_name, [])
            qm = qm_fields[field_name]
            field_comment = qm.get("comment", "")
            field_type = qm.get("type", "float")

            # Get output_fields description from first writer module
            out_desc = ""
            for r in results:
                if field_name in r["output_fields"]:
                    out_desc = r["output_fields"][field_name]
                    break

            direction = _get_score_direction(field_name, out_desc or field_comment)

            # Extract range, stripping direction keywords to avoid duplication
            range_str = ""
            range_src = out_desc or field_comment
            if range_src:
                range_match = re.search(
                    r"\(([^)]*(?:higher|lower|dB|MOS|0-\d+|\d+-\d+)[^)]*)\)", range_src
                )
                if range_match:
                    raw = range_match.group(1)
                    # Strip direction text to avoid "↑ higher=better · higher=better"
                    raw = re.sub(r",?\s*(?:higher|lower)\s*=\s*better", "", raw).strip(" ,")
                    range_str = raw
                else:
                    # Try bare range like "0-10" or "1-5"
                    bare = re.search(r"(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)", range_src)
                    if bare:
                        range_str = bare.group(1)

            # Build tagline: field comment (or module desc) + direction + range
            tagline_parts = []
            if field_comment:
                tagline_parts.append(field_comment)
            if direction != "—":
                tagline_parts.append(direction)
            if range_str:
                tagline_parts.append(range_str)
            if field_type != "float":
                tagline_parts.append(f"type: {field_type}")
            tagline = " · ".join(tagline_parts)

            a(f"### `{field_name}`")
            if tagline:
                a(f"> {tagline}")
            a("")

            # Show which modules read this field
            readers = field_reader_names.get(field_name, [])
            if readers:
                reader_links = []
                for rn in sorted(readers):
                    rextra = mod_extra.get(rn, {})
                    rpath = rextra.get("file_path", "")
                    reader_links.append(f"[`{rn}`]({rpath})" if rpath else f"`{rn}`")
                a(f"Used by: {', '.join(reader_links)}")
                a("")

            # One info block per writer module
            for mod_name in sorted(writers):
                mod = mod_lookup.get(mod_name)
                if mod is None:
                    continue
                extra = mod_extra.get(mod_name, {})
                file_path = extra.get("file_path", "")
                source_links = extra.get("source_links", "—")
                chain = " → ".join(mod["fallback_chain"]) if mod["fallback_chain"] else ""
                badge = speed_badges.get(mod["speed"], "")
                test_status = _format_test_status(mod_name, test_results)
                pkgs = ", ".join(mod["packages"]) if mod["packages"] else ""
                vram = mod["vram"] or ""

                if file_path:
                    mod_link = f"[`{mod_name}`]({file_path})"
                else:
                    mod_link = f"`{mod_name}`"

                a(f"**{mod_link}** — {mod['description']}")
                a("")

                # Info as bullet list
                speed_str = f"{badge} {mod['speed']}"
                if mod["gpu"]:
                    speed_str += " · GPU"
                a(f"- **Input**: {mod['input_type']} · **Speed**: {speed_str}")
                if chain:
                    a(f"- **Backend**: {chain}")
                if pkgs:
                    a(f"- **Packages**: {pkgs}")
                if vram:
                    a(f"- **VRAM**: {vram}")
                if source_links != "—":
                    a(f"- **Source**: {source_links}")
                if test_status != "\u2014":
                    a(f"- **Test**: {test_status}")

                cfg = mod.get("default_config", {})
                cfg_items = {k: v for k, v in cfg.items()
                             if k not in ("weights_path", "preferred_backend", "models_dir")
                             and v is not None}
                if cfg_items:
                    cfg_str = ", ".join(f"`{k}={v}`" for k, v in cfg_items.items())
                    a(f"- **Config**: {cfg_str}")
                a("")

        a("")

    # ── Utility & Validation Modules (no metric output) ───────────────
    no_output = [r for r in results if not r["output_fields"]]
    if no_output:
        a(f"## Utility & Validation ({len(no_output)} modules)")
        a("")
        a("Modules that perform validation, embedding, deduplication, or dataset-level "
          "analysis without writing individual QualityMetrics fields.")
        a("")
        for r in sorted(no_output, key=lambda x: x["name"]):
            extra = mod_extra.get(r["name"], {})
            file_path = extra.get("file_path", "")
            mod_link = f"[`{r['name']}`]({file_path})" if file_path else f"`{r['name']}`"
            badge = speed_badges.get(r["speed"], "")
            items = [f"Input: {r['input_type']}", f"Speed: {badge} {r['speed']}"]
            if r["gpu"]:
                items.append("GPU")
            a(f"- **{mod_link}** — {r['description']} · {' · '.join(items)}")
        a("")

    return "\n".join(L)
