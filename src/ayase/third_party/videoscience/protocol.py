"""VideoScience-Judge prompt, parsing, scoring, and frame protocol.

Adapted from hao-ai-lab/VideoScience at the commit recorded in NOTICE.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOURCE_REVISION = "0b3bb2da74388991bf842bca1096a6a25d3c9311"
DATASET_REVISION = "a5e038f3b2a65843f935d883c6ba269d304ff769"

WEIGHTS = {
    "prompt_consistency": 0.20,
    "expected_phenomenon": 0.30,
    "coherence": 0.20,
    "immutability": 0.15,
    "dynamism": 0.15,
}
CHECKLIST_KEYS = (
    "phenomenon_congruency",
    "correct_dynamism",
    "spatio_temporal_continuity",
    "immutability",
    "interaction_realism",
)

RUBRIC_TEXT = (
    "You are VLM-Judge evaluating a generated science video.\n"
    "Score each rubric from 1–4 (1=absent/contradictory, 2=weak/partly wrong, 3=mostly correct, 4=clearly correct):\n"
    "a) prompt_consistency — follows instructions: correct setup and correct experiment execution.\n"
    "b) expected_phenomenon — expected physical/chemical outcome is present and correct.\n"
    "c) immutability — objects remain intact/unchanged unless changes are explicitly expected.\n"
    "d) dynamism — other physical laws are obeyed.\n"
    "e) coherence — natural transitions across frames; no flicker/teleport/identity swap.\n"
)
SCHEMA_HINT = (
    "Return JSON with fields:\n"
    '{ "scores": {\n'
    '    "prompt_consistency":1-4,\n'
    '    "expected_phenomenon":1-4,\n'
    '    "immutability":1-4,\n'
    '    "dynamism":1-4,\n'
    '    "coherence":1-4\n'
    "  },\n"
    '  "explanations": {"summary": string, "issues": [string]},\n'
    '  "evidence": {"candidate": [{"t":"0.0s","observation":""}],'
    '               "reference": [{"t":"0.0s","observation":""}]}\n'
    "}\n"
)


def format_checklist(checklist: Dict[str, List[str]]) -> str:
    headings = {
        "phenomenon_congruency": "PHENOMENON CONGRUENCY",
        "correct_dynamism": "CORRECT DYNAMISM",
        "spatio_temporal_continuity": "SPATIO-TEMPORAL CONTINUITY",
        "immutability": "IMMUTABILITY",
        "interaction_realism": "INTERACTION REALISM",
    }
    lines = [
        "",
        "Expected phenomenon checklist (each bullet describes what SHOULD be observable in the video):",
    ]
    for key, heading in headings.items():
        items = checklist.get(key) or []
        if items:
            lines.append(f"\n{heading}:")
            lines.extend(f"- {item.strip()}" for item in items)
    lines.append(
        "\nUse this checklist as ground truth for what a correct video should show. "
        "If many items for a category are violated, that category's score should be low. "
        "If almost all items are satisfied, that category's score should be high."
    )
    return "\n".join(lines)


def _format_cv_metrics(cv_metrics: Dict[str, Any]) -> str:
    results = cv_metrics.get("results") or {}
    if not isinstance(results, dict):
        results = {}
    lines = ["", "Computer-vision analysis (noisy; use as secondary evidence, not ground truth):"]

    def number(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    grounding = results.get("grounding_dino") or results.get("GroundingDINO")
    if (
        isinstance(grounding, dict)
        and number(grounding.get("frames_with_any_detection_ratio")) is not None
    ):
        ratio = number(grounding["frames_with_any_detection_ratio"])
        lines.append(
            f"- GroundingDINO: target entities detected in ~{ratio * 100:.1f}% of analyzed frames."
        )
    tracking = results.get("bytetrack")
    if isinstance(tracking, dict):
        pieces = []
        coherence = number(tracking.get("coherence_score"))
        average = number(tracking.get("avg_track_length_frames"))
        if coherence is not None:
            pieces.append(f"coherence_score≈{coherence:.2f} (1.0 means long, stable tracks)")
        if isinstance(tracking.get("num_tracks"), (int, float)):
            pieces.append(f"{int(tracking['num_tracks'])} active tracks")
        if average is not None:
            pieces.append(f"avg track length≈{average:.1f} frames")
        if pieces:
            lines.append("- ByteTrack: " + ", ".join(pieces) + ".")
    flow = results.get("raft")
    if isinstance(flow, dict) and number(flow.get("mean_flow_magnitude")) is not None:
        magnitude = number(flow["mean_flow_magnitude"])
        angle = number(flow.get("mean_flow_direction_degrees"))
        suffix = f", mean direction≈{angle:.1f}° (image coords)." if angle is not None else "."
        lines.append(f"- RAFT optical flow: mean flow magnitude≈{magnitude:.3f}" + suffix)
    clip = results.get("clip4clip") or results.get("CLIP4Clip")
    if isinstance(clip, dict) and number(clip.get("similarity_cosine")) is not None:
        similarity = number(clip["similarity_cosine"])
        lines.append(
            f"- CLIP4Clip text–video alignment: cosine similarity≈{similarity:.3f} (higher generally means better alignment of video with the prompt)."
        )
    lpips = results.get("lpips")
    if isinstance(lpips, dict) and number(lpips.get("lpips_mean")) is not None:
        mean = number(lpips["lpips_mean"])
        lines.append(
            f"- LPIPS (VGG, candidate vs ref): mean LPIPS≈{mean:.3f} (lower means more perceptually similar to the reference video)."
        )
    if len(lines) == 2:
        return ""
    lines.append(
        "\nUse these signals only as hints: if they strongly contradict what you see in the frames, trust the actual video content over these metrics."
    )
    return "\n".join(lines)


def build_judge_prompt(
    phenomenon: str,
    description: str,
    checklist: Optional[Dict[str, List[str]]] = None,
    cv_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = (
        RUBRIC_TEXT
        + "\n"
        + SCHEMA_HINT
        + "\n"
        + f"Ground-truth phenomenon: {phenomenon}\n\n"
        + "Ground-truth description (authoritative):\n"
        + description.strip()
    )
    if checklist:
        prompt += "\n\n" + format_checklist(checklist)
    if cv_metrics:
        prompt += "\n\n" + _format_cv_metrics(cv_metrics)
    return prompt


def build_checklist_prompt(
    prompt_text: str,
    reference_phenomenon: str = "",
    additional_reference_from_multimedia: str = "",
    source_link: str = "",
) -> str:
    return f"""From my evaluation of text2video models I have generated a video using the prompt:

Prompt: {prompt_text}

Reference ground-truth phenomenon (expected outcome / behavior):
{reference_phenomenon}

Additional information about the multimedia or reference sources:
{additional_reference_from_multimedia}

Reference video or page URL (if available):
{source_link}

Use these reference sources to create the checklist (and the reference video if available).

Create a checklist targeting each of the following categories (note: not all categories are required for a prompt):

1. PHENOMENON CONGRUENCY
   - Does the video show the correct expected phenomenon?

2. CORRECT DYNAMISM
   - Are the physics dynamics and motion behaviors accurate?

3. SPATIO-TEMPORAL CONTINUITY
   - Are spatial relationships and temporal sequences physically consistent?

4. IMMUTABILITY
   - Do object properties remain physically consistent?

5. INTERACTION REALISM
   - Do object interactions follow physical laws (including those not explicitly described in the expected phenomenon)?

Guidelines for checklist creation:
- Only target things which are visually observable in the video.
- The statements in the checklist must be assertive statements that can be answered with YES or NO (not questions).
- Each statement should be as concrete and testable as possible.
"""


def _json_object(text: str) -> Dict[str, Any]:
    candidates = [text]
    candidates.extend(
        m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    )
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")


def parse_judgment(text: str) -> Dict[str, Any]:
    data = _json_object(text)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        scores = data.get("rubric")
    if not isinstance(scores, dict):
        raise ValueError("response has no scores object")
    parsed: Dict[str, float] = {}
    for key in WEIGHTS:
        value = scores.get(key)
        if isinstance(value, bool):
            raise ValueError(f"{key} is not numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} is not numeric") from exc
        if not math.isfinite(number) or not 1.0 <= number <= 4.0:
            raise ValueError(f"{key} must be finite and within 1..4")
        parsed[key] = number
    return {
        "scores": parsed,
        "explanations": data.get("explanations", {}),
        "evidence": data.get("evidence", {}),
    }


def parse_checklist(text: str) -> Dict[str, List[str]]:
    data = _json_object(text)
    if set(data) != set(CHECKLIST_KEYS):
        raise ValueError("checklist must contain exactly the five released keys")
    result: Dict[str, List[str]] = {}
    for key in CHECKLIST_KEYS:
        items = data[key]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"invalid checklist category: {key}")
        result[key] = [item.strip() for item in items]
    return result


def compute_overall(scores: Dict[str, float]) -> float:
    return sum(scores[key] * weight for key, weight in WEIGHTS.items())


def extract_frames(path: Path, count: int) -> List[Tuple[float, bytes]]:
    """Extract source-compatible timestamps as JPEG quality 80 byte strings."""
    import cv2
    from PIL import Image

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "JPEG", quality=80)
        return [(0.0, buffer.getvalue())]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / max(fps, 1e-6)
    frames: List[Tuple[float, bytes]] = []
    try:
        for index in range(max(1, count)):
            timestamp = index * duration / max(1, count)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, "JPEG", quality=80)
            frames.append((timestamp, buffer.getvalue()))
    finally:
        cap.release()
    return frames


def data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
