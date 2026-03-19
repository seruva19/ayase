"""Generate METRICS.md from module metadata via PipelineModule.get_metadata()."""

from collections import Counter
from typing import Dict, List

from .pipeline import ModuleRegistry, PipelineModule


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


def generate_metrics_doc() -> str:
    all_modules = ModuleRegistry.list_modules()

    results: List[Dict] = []
    for name in all_modules:
        cls = ModuleRegistry.get_module(name)
        if cls is None or cls.name == "unnamed_module":
            continue
        meta = cls.get_metadata()
        meta["group"] = _get_group(meta["name"], meta["input_type"])
        results.append(meta)

    results.sort(key=lambda x: (x["group"], x["name"]))

    total_outputs = sum(len(r["output_fields"]) for r in results)
    lines = [
        "# Ayase Metrics Reference",
        "",
        f"{len(results)} modules, {total_outputs} output fields.",
        "",
    ]

    current_group = None
    for r in results:
        if r["group"] != current_group:
            if current_group is not None:
                lines.append("")
            lines.append(f"## {r['group']}")
            lines.append("")
            lines.append("| Module | Input | Outputs | Description | Config |")
            lines.append("|--------|-------|---------|-------------|--------|")
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

        lines.append(
            f"| `{r['name']}` | {r['input_type']} | {out_str} "
            f"| {r['description']} | {cfg_str} |"
        )

    lines.append("")
    return "\n".join(lines)
