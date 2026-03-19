"""Generate METRICS.md from module metadata (no source parsing — uses PipelineModule.get_metadata())."""

import importlib
import pkgutil
from collections import Counter
from pathlib import Path

from ayase.pipeline import PipelineModule

MODULES_DIR = Path(__file__).resolve().parent.parent / "src" / "ayase" / "modules"
OUTPUT = Path(__file__).resolve().parent.parent / "METRICS.md"


def get_group(name: str, input_type: str) -> str:
    """Assign functional group based on module name and input type."""
    if input_type.startswith("audio"):
        return "Audio Quality"
    if input_type == "batch":
        return "Distribution (batch)"

    motion_kw = (
        "motion", "temporal", "flicker", "flow", "subject_consist",
        "background_consist", "scene_detect", "jump_cut", "playback",
        "camera_jitter", "camera_motion", "flow_coherence",
        "object_permanence", "stabilized", "warping", "raft_motion",
        "ptlflow", "judder", "vfr",
    )
    if any(x in name for x in motion_kw):
        return "Motion & Temporal"

    vqa_kw = (
        "dover", "fast_vqa", "mdtvsfa", "videval", "tlvqm", "c3dvqa",
        "cover", "finevq", "kvq", "rqvqa", "funque", "st_greed",
        "hdr_vqm", "cgvqm", "movie",
    )
    if any(x in name for x in vqa_kw):
        return "Video Quality Assessment"

    vgen_kw = (
        "videoscore", "video_reward", "aigv", "chronomagic",
        "t2v_comp", "video_type", "video_memor", "t2v_score",
    )
    if any(x in name for x in vgen_kw):
        return "Video Generation"

    if any(x in name for x in ("av_sync", "audio_visual")):
        return "Audio-Visual"

    fr_kw = (
        "vmaf", "ssimulacra", "butteraugli", "flip", "psnr", "ssim",
        "ciede", "pieapp", "cw_ssim", "nlpd", "ahiq", "topiq_fr",
        "dreamsim", "dmm", "wadiqam_fr", "ssimc", "xpsnr", "hdr_vdp",
        "delta_ictcp", "ckdn", "deepwsd", "strred", "flolpips",
        "st_lpips", "vif", "fvd", "fvmd", "kvd", "mad",
    )
    if any(x in name for x in fr_kw):
        return "Full-Reference & Distribution"

    if any(x in name for x in ("hdr_", "pu_metric", "tonal")):
        return "HDR & Color"
    if any(x in name for x in ("nsfw", "harmful", "deepfake", "watermark", "bias")):
        return "Safety & Content"
    if "face" in name or "identity" in name:
        return "Face & Identity"

    sem_kw = (
        "semantic", "caption", "clip_temp", "clip_iqa", "sd_reference",
        "ocr", "vqa_score", "tifa", "nemo", "text_", "video_text",
        "ram_tag", "promptiqa",
    )
    if any(x in name for x in sem_kw):
        return "Text & Semantic"

    if "depth" in name:
        return "Depth"
    if "i2v" in name:
        return "Image-to-Video Reference"
    if any(x in name for x in ("codec", "compression", "letterbox")):
        return "Codec & Technical"
    return "No-Reference Quality"


def main() -> None:
    results = []
    for finder, mod_name, ispkg in pkgutil.iter_modules([str(MODULES_DIR)]):
        if mod_name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"ayase.modules.{mod_name}")
        except Exception:
            continue

        for attr in sorted(dir(mod)):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, PipelineModule)
                and obj is not PipelineModule
                and obj.name != "unnamed_module"
            ):
                meta = obj.get_metadata()
                meta["group"] = get_group(meta["name"], meta["input_type"])
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
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    groups = Counter(r["group"] for r in results)
    print(f"Total: {len(results)} modules, {total_outputs} outputs")
    for g, c in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
