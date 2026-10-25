"""VGIF-Score QA and AutoRubric protocol primitives.

Adapted from PRIS-CV/VGIF-SCORE commit
``b7558fe27bb13b97cd6039dfa421da4d11c4f2bc`` (Apache-2.0).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable


DIMENSIONS = (
    ("Cin", "cinematography", "Cinematography"),
    ("Pur", "purity", "Purity"),
    ("Mot", "motion_smoothness", "Motion smoothness"),
    ("Phy", "physics_adherence", "Physics adherence"),
)


def dependency_ids(expression: str | None) -> list[str]:
    if not expression or expression == "None":
        return []
    return re.findall(r"q\d+", expression)


def _questions_blob(qa_pairs: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'{qa["id"]} | type={qa.get("type")} | dependency={qa.get("dependency")} | question={qa["question"]}'
        for qa in qa_pairs
    )


def build_dependency_round_prompt(
    *,
    round_index: int,
    total_rounds: int,
    current_round_qas: list[dict[str, Any]],
    locked_answers: dict[str, dict[str, str]],
) -> str:
    previous_lines = [
        f"{qa_id} = {payload.get('answer', 'UNKNOWN')} | reason={payload.get('reason', '')}"
        for qa_id, payload in sorted(locked_answers.items())
    ]
    previous_blob = "\n".join(previous_lines) if previous_lines else "None"
    questions_blob = _questions_blob(current_round_qas)
    return (
        "You are a strict video QA auditor for text-to-video evaluation.\n"
        "Inspect the video itself carefully and answer only the current round questions.\n"
        "Do not revise previous rounds. Previous rounds are locked context only.\n"
        "If evidence is uncertain, partially visible, implied, off-screen, or ambiguous, answer No.\n"
        "\n"
        f"This is round {round_index} of {total_rounds}.\n"
        "Previous locked answers:\n"
        f"{previous_blob}\n"
        "\n"
        "Return JSON only, with this exact schema:\n"
        '{"answers":[{"id":"q1","answer":"Yes","reason":"short reason"}]}\n'
        "You must include every current-round question id exactly once.\n"
        "Use only Yes or No.\n"
        "Keep each reason under 18 words and mention brief visual evidence.\n"
        "Current round questions:\n"
        f"{questions_blob}"
    )


def _dimension_payload(annotations: dict[str, Any], key: str) -> dict[str, Any]:
    dimensions = annotations["dimensions"]
    payload = dimensions.get(key)
    if key == "purity" and payload is None:
        payload = dimensions.get("visual_purity")
    return payload


def build_autorubric_prompt(annotations: dict[str, Any]) -> str:
    blocks = []
    for code, key, display_name in DIMENSIONS:
        dimension = _dimension_payload(annotations, key)
        anchors = dimension["score_anchors"]
        anchors_blob = "\n".join(
            f"    {score}: {text}"
            for score, text in sorted(anchors.items(), key=lambda item: int(item[0]))
        )
        focus = dimension["focus_points"]
        focus_blob = ", ".join(focus) if isinstance(focus, list) else str(focus)
        blocks.append(
            f"{code} ({display_name})\n"
            f"  Goal: {dimension['dimension_goal']}\n"
            f"  Scoring prompt: {dimension['vlm_scoring_prompt']}\n"
            f"  Focus points: {focus_blob}\n"
            "  Score anchors:\n"
            f"{anchors_blob}"
        )
    qa_checklist = "\n".join(
        f'{qa["id"]} | dependency={qa.get("dependency")} | {qa["question"]}'
        for qa in annotations["original_qa_pairs"]
    )
    dimension_blob = "\n\n".join(blocks)
    return (
        "You are a strict video auto-rubric evaluator for text-to-video generation.\n"
        "Inspect the video carefully and assign integer scores from 1 to 5 for each requested dimension.\n"
        "Use only visible evidence in the video. Do not give credit for events that are only implied by the prompt.\n"
        "Be conservative: if evidence is ambiguous, lower the score.\n"
        "\n"
        "Original text-to-video prompt:\n"
        f"{annotations['prompt']}\n"
        "\n"
        "QA checklist from the dataset. Use this as supporting context for prompt-specific requirements:\n"
        f"{qa_checklist}\n"
        "\n"
        "Score these four dimensions:\n"
        f"{dimension_blob}\n\n"
        "\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "scores": {\n'
        '    "Cin": {"score": 1, "reason": "short visual reason"},\n'
        '    "Pur": {"score": 1, "reason": "short visual reason"},\n'
        '    "Mot": {"score": 1, "reason": "short visual reason"},\n'
        '    "Phy": {"score": 1, "reason": "short visual reason"}\n'
        "  },\n"
        '  "summary": "one sentence summary"\n'
        "}\n"
        "Every score must be an integer 1, 2, 3, 4, or 5. Keep each reason under 24 words."
    )


def _parse_dependency(expression: str | None, predicate: Callable[[str], bool]) -> tuple[bool, list[str]]:
    if not expression or expression == "None":
        return True, []
    tokens = re.findall(r"q\d+|AND|OR|\(|\)", expression, flags=re.IGNORECASE)
    normalized = [token.upper() if token.upper() in {"AND", "OR", "(", ")"} else token for token in tokens]
    position = 0

    def parse_or() -> bool:
        nonlocal position
        value = parse_and()
        while position < len(normalized) and normalized[position] == "OR":
            position += 1
            rhs = parse_and()
            value = value or rhs
        return value

    def parse_and() -> bool:
        nonlocal position
        value = parse_primary()
        while position < len(normalized) and normalized[position] == "AND":
            position += 1
            rhs = parse_primary()
            value = value and rhs
        return value

    def parse_primary() -> bool:
        nonlocal position
        if position >= len(normalized):
            raise ValueError("incomplete dependency expression")
        token = normalized[position]
        if token == "(":
            position += 1
            value = parse_or()
            if position >= len(normalized) or normalized[position] != ")":
                raise ValueError("unmatched dependency parenthesis")
            position += 1
            return value
        if re.fullmatch(r"q\d+", token):
            position += 1
            return predicate(token)
        raise ValueError(f"invalid dependency token: {token}")

    passed = parse_or()
    if position != len(normalized):
        raise ValueError("extra dependency tokens")
    return passed, [qa_id for qa_id in dependency_ids(expression) if not predicate(qa_id)]


def validate_annotations(annotations: dict[str, Any]) -> None:
    if not isinstance(annotations, dict):
        raise ValueError("vgif_annotations must be an object")
    if not isinstance(annotations.get("prompt"), str) or not annotations["prompt"].strip():
        raise ValueError("VGIF original prompt is required")
    qas = annotations.get("original_qa_pairs")
    if not isinstance(qas, list) or not qas:
        raise ValueError("VGIF original_qa_pairs must be non-empty")
    ids = []
    for qa in qas:
        if not isinstance(qa, dict) or not isinstance(qa.get("id"), str) or not re.fullmatch(r"q\d+", qa["id"]):
            raise ValueError("each VGIF QA requires a q<number> id")
        if not isinstance(qa.get("question"), str) or not qa["question"].strip():
            raise ValueError(f"VGIF QA {qa['id']} has no question")
        if str(qa.get("expected_answer", "")).strip().lower() not in {"yes", "no"}:
            raise ValueError(f"VGIF QA {qa['id']} requires a Yes/No expected_answer")
        ids.append(qa["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate VGIF QA id")
    known = set(ids)
    for qa in qas:
        expression = qa.get("dependency")
        if expression and expression != "None":
            compact = re.sub(r"q\d+|AND|OR|\(|\)|\s+", "", str(expression), flags=re.IGNORECASE)
            if compact:
                raise ValueError(f"invalid dependency grammar for {qa['id']}")
            missing = set(dependency_ids(str(expression))) - known
            if missing:
                raise ValueError(f"missing dependency for {qa['id']}: {sorted(missing)}")
            _parse_dependency(str(expression), lambda _: True)
    build_dependency_rounds(qas)
    dimensions = annotations.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("VGIF rubric dimensions are required")
    for _, key, _ in DIMENSIONS:
        payload = dimensions.get(key, dimensions.get("visual_purity") if key == "purity" else None)
        if not isinstance(payload, dict):
            raise ValueError(f"missing VGIF rubric dimension: {key}")
        for field in ("dimension_goal", "vlm_scoring_prompt", "focus_points", "score_anchors"):
            if field not in payload:
                raise ValueError(f"VGIF rubric {key} missing {field}")
        if not isinstance(payload["dimension_goal"], str) or not payload["dimension_goal"].strip():
            raise ValueError(f"VGIF rubric {key} has an empty dimension_goal")
        if not isinstance(payload["vlm_scoring_prompt"], str) or not payload["vlm_scoring_prompt"].strip():
            raise ValueError(f"VGIF rubric {key} has an empty vlm_scoring_prompt")
        focus = payload["focus_points"]
        if not isinstance(focus, list) or not focus or not all(isinstance(item, str) and item.strip() for item in focus):
            raise ValueError(f"VGIF rubric {key} requires non-empty focus_points")
        anchors = payload["score_anchors"]
        if not isinstance(anchors, dict) or {str(key) for key in anchors} != {"1", "2", "3", "4", "5"}:
            raise ValueError(f"VGIF rubric {key} requires anchors 1..5")
        if not all(isinstance(value, str) and value.strip() for value in anchors.values()):
            raise ValueError(f"VGIF rubric {key} has an empty score anchor")


def build_dependency_rounds(qa_pairs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_id = {qa["id"]: qa for qa in qa_pairs}
    visiting: set[str] = set()
    cache: dict[str, int] = {}

    def level(qa_id: str) -> int:
        if qa_id in cache:
            return cache[qa_id]
        if qa_id in visiting:
            raise ValueError(f"dependency cycle detected at {qa_id}")
        visiting.add(qa_id)
        deps = dependency_ids(by_id[qa_id].get("dependency"))
        value = max((level(dep) for dep in deps), default=-1) + 1
        visiting.remove(qa_id)
        cache[qa_id] = value
        return value

    grouped: dict[int, list[dict[str, Any]]] = {}
    for qa in qa_pairs:
        grouped.setdefault(level(qa["id"]), []).append(qa)
    return [grouped[index] for index in sorted(grouped)]


def evaluate_predictions(qa_pairs: list[dict[str, Any]], answers: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], int, int, int]:
    rows = []
    by_id = {}
    for qa in qa_pairs:
        prediction = answers.get(qa["id"], {"answer": "UNKNOWN", "reason": ""})
        expected = str(qa["expected_answer"]).strip().title()
        row = {
            "id": qa["id"], "question": qa["question"], "type": qa.get("type"),
            "dependency": qa.get("dependency"), "expected_answer": expected,
            "predicted_answer": prediction["answer"], "answer_match": prediction["answer"] == expected,
            "dependency_passed": True, "dependency_failed_ids": [], "correct": False,
            "reason": prediction.get("reason", ""),
        }
        rows.append(row)
        by_id[qa["id"]] = row
    visiting: set[str] = set()

    @lru_cache(maxsize=None)
    def correct(qa_id: str) -> bool:
        if qa_id in visiting:
            raise ValueError(f"dependency cycle detected at {qa_id}")
        row = by_id.get(qa_id)
        if row is None:
            return False
        visiting.add(qa_id)
        try:
            passed, failed = _parse_dependency(row["dependency"], correct)
            row["dependency_passed"] = passed
            row["dependency_failed_ids"] = failed
            row["correct"] = bool(row["answer_match"] and passed)
            return row["correct"]
        finally:
            visiting.remove(qa_id)

    for qa_id in by_id:
        correct(qa_id)
    raw = sum(row["answer_match"] for row in rows)
    gated = sum(row["correct"] for row in rows)
    blocked = sum(row["answer_match"] and not row["dependency_passed"] for row in rows)
    return rows, raw, gated, blocked
