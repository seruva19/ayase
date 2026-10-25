"""MIT-licensed protocol adapter for VideoScience-Judge."""

from .protocol import (
    CHECKLIST_KEYS,
    WEIGHTS,
    build_checklist_prompt,
    build_judge_prompt,
    compute_overall,
    extract_frames,
    format_checklist,
    parse_checklist,
    parse_judgment,
)

__all__ = [
    "CHECKLIST_KEYS",
    "WEIGHTS",
    "build_checklist_prompt",
    "build_judge_prompt",
    "compute_overall",
    "extract_frames",
    "format_checklist",
    "parse_checklist",
    "parse_judgment",
]
