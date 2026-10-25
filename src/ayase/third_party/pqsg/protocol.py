"""PQSG prompt generation, video QA, validation, and dependency scoring.

Adapted from atin pothiraj/pqsg at commit 03afdb0e6f9ed68d363a68eea5ca4fd90d53848e.
"""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROMPT_PATH = Path(__file__).with_name("qg_prompt.txt")
CATEGORIES = ("object_existence", "action_verification", "physics")
PREFIX = {"object_existence": "O", "action_verification": "A", "physics": "P"}
FORWARD_EDGES = frozenset({("O", "A"), ("O", "P"), ("A", "A"), ("A", "P"), ("P", "P")})


def _extract_json(text: str) -> str:
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
    return text


def graph_prompt(prompt: str) -> str:
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\nNow produce a similar JSON for this new prompt and reference video:\n\n"
        + f"Prompt: {prompt}"
    )


def _response_id(response: Any) -> Optional[str]:
    value = getattr(response, "id", None) or getattr(response, "response_id", None)
    return str(value) if value is not None else None


def _response_model(response: Any) -> Optional[str]:
    value = getattr(response, "model", None) or getattr(response, "model_version", None)
    return str(value) if value is not None else None


def generate_psg(
    prompt: str,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
) -> Tuple[dict, dict]:
    """Generate a fresh question graph without sending the video."""
    full_prompt = graph_prompt(prompt)
    if backend == "gemini":
        if client is None:
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("PQSG Gemini requires GOOGLE_API_KEY")
            from google import genai

            client = genai.Client(api_key=api_key)
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = response.text or ""
    elif backend == "openai":
        if client is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("PQSG OpenAI requires OPENAI_API_KEY")
            from openai import OpenAI

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates structured JSON output for video evaluation questions.",
                },
                {"role": "user", "content": full_prompt},
            ],
            max_completion_tokens=16384,
        )
        text = response.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported PQSG backend: {backend}")
    if not text.strip():
        raise RuntimeError("PQSG question generation returned an empty response")
    return json.loads(_extract_json(text)), {
        "requested_model": model,
        "returned_model": _response_model(response),
        "request_id": _response_id(response),
        "raw_response": text,
    }


def validate_graph(graph: dict) -> None:
    """Reject malformed, empty, dangling, backward, and cyclic graphs."""
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), dict):
        raise ValueError("PQSG graph must contain a nodes object")
    nodes = graph["nodes"]
    if tuple(nodes) != CATEGORIES:
        raise ValueError("PQSG categories must use released order")
    ids: Dict[str, str] = {}
    for category in CATEGORIES:
        questions = nodes[category]
        if not isinstance(questions, dict):
            raise ValueError(f"PQSG category {category} must be an object")
        for qid, question in questions.items():
            if not isinstance(qid, str) or not qid.startswith(PREFIX[category]):
                raise ValueError(f"Invalid PQSG node ID: {qid!r}")
            if qid in ids or not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid or duplicate PQSG node: {qid!r}")
            ids[qid] = category
    if not ids:
        raise ValueError("PQSG graph contains no questions")
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("PQSG edges must be a list")
    children: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("from"), list) or not edge["from"]:
            raise ValueError("Malformed PQSG edge")
        target = edge.get("to")
        if target not in ids:
            raise ValueError(f"Dangling PQSG target: {target!r}")
        for parent in edge["from"]:
            if parent not in ids:
                raise ValueError(f"Dangling PQSG parent: {parent!r}")
            if (parent[0], target[0]) not in FORWARD_EDGES:
                raise ValueError(f"Unsupported PQSG edge: {parent}->{target}")
            children[parent].append(target)
    visiting, visited = set(), set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("PQSG graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in children.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in ids:
        visit(node)


def _flatten(graph: dict) -> Tuple[List[str], Dict[str, dict]]:
    order, mapping = [], {}
    for category in CATEGORIES:
        for qid, question in graph["nodes"][category].items():
            order.append(qid)
            mapping[qid] = {"text": question, "category": category}
    return order, mapping


def _parse_initial(text: str, ids: List[str]) -> Dict[str, str]:
    out = {qid: "" for qid in ids}
    current = None
    lines: List[str] = []
    for line in text.split("\n"):
        found = next(
            (qid for qid in ids if f"[{qid}]" in line or line.strip().startswith(f"{qid}:")), None
        )
        if found:
            if current:
                out[current] = "\n".join(lines).strip()
            current, lines = found, [line]
        elif current:
            lines.append(line)
    if current:
        out[current] = "\n".join(lines).strip()
    return out


def _parse_yesno(text: str, ids: List[str]) -> Dict[str, str]:
    out = {qid: "no" for qid in ids}
    for line in text.lower().split("\n"):
        line = line.strip()
        for qid in ids:
            if f"[{qid.lower()}]" in line or line.startswith(f"{qid.lower()}:"):
                if ("yes" in line and "no" not in line) or line.endswith("yes"):
                    out[qid] = "yes"
                break
    return out


def _answer_coverage(text: str, ids: List[str]) -> float:
    labeled = {
        qid
        for line in text.lower().split("\n")
        for qid in ids
        if f"[{qid.lower()}]" in line or line.strip().startswith(f"{qid.lower()}:")
    }
    return len(labeled) / len(ids) if ids else 0.0


def _answers(ids: List[str], mapping: dict, initial_text: str, final_text: str) -> dict:
    initial, final = _parse_initial(initial_text, ids), _parse_yesno(final_text, ids)
    result: dict = {}
    for qid in ids:
        yes = final[qid] == "yes"
        category = mapping[qid]["category"]
        result.setdefault(category, {})[qid] = {
            "question": mapping[qid]["text"],
            "initial_response": initial[qid],
            "final_answer": final[qid],
            "is_correct": yes,
            "score": int(yes),
        }
    return result


def _prompts(graph: dict) -> Tuple[List[str], dict, str, str, str]:
    ids, mapping = _flatten(graph)
    questions = "".join(f"{i + 1}. [{qid}] {mapping[qid]['text']}\n" for i, qid in enumerate(ids))
    gemini = (
        "You are a verifier for AI-generated videos. Given a list of questions and a video, verify whether the video satisfies each question.\n\n"
        f"QUESTIONS:\n{questions}\n"
        "Provide your answer for each question in the format:\n[Question ID]: Your detailed reasoning here\n\nAnswer each question completely."
    )
    openai = (
        "You are a verifier for AI-generated videos. Given a list of questions and a sequence of video frames, verify whether the video satisfies each question.\n\n"
        f"QUESTIONS:\n{questions}\n"
        "Format: [Question ID]: detailed reasoning. Answer each question completely."
    )
    return ids, mapping, gemini, openai, questions


def answer_gemini(
    video_path: str,
    graph: dict,
    model: str,
    fps: int,
    api_key: Optional[str] = None,
    client: Any = None,
) -> Tuple[dict, dict]:
    if client is None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("PQSG Gemini requires GOOGLE_API_KEY")
        from google import genai

        client = genai.Client(api_key=api_key)
    from google.genai import types

    ids, mapping, step1, _, _ = _prompts(graph)
    mime = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
    }.get(Path(video_path).suffix.lower(), "video/mp4")
    video = types.Part(
        inline_data=types.Blob(data=Path(video_path).read_bytes(), mime_type=mime),
        video_metadata=types.VideoMetadata(fps=fps),
    )
    r1 = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[video, types.Part(text=step1)]),
        config=types.GenerateContentConfig(temperature=0.0),
    )
    initial = "".join(
        p.text for p in r1.candidates[0].content.parts if getattr(p, "text", None)
    ).strip()
    followup = f"Based on your previous answers, give a yes/no for each question.\nRespond ONE per line:\n[Question ID]: yes  or  [Question ID]: no\nProvide answers for all {len(ids)} questions."
    r2 = client.models.generate_content(
        model=model,
        contents=types.Content(
            parts=[
                video,
                types.Part(text=step1),
                types.Part(text=f"Assistant response:\n{initial}"),
                types.Part(text=followup),
            ]
        ),
        config=types.GenerateContentConfig(temperature=0.0),
    )
    final = "".join(
        p.text for p in r2.candidates[0].content.parts if getattr(p, "text", None)
    ).strip()
    return _answers(ids, mapping, initial, final), {
        "requested_model": model,
        "returned_models": [_response_model(r1), _response_model(r2)],
        "request_ids": [_response_id(r1), _response_id(r2)],
        "raw_responses": [initial, final],
        "coverage": _answer_coverage(final, ids),
    }


def _extract_frames(video_path: str, fps: int, jpeg_quality: int = 70) -> List[str]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(native_fps / fps))
    frames = []
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % interval == 0:
                encoded, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if encoded:
                    frames.append(base64.b64encode(buf).decode("utf-8"))
            count += 1
    finally:
        cap.release()
    return frames


def answer_openai(
    video_path: str,
    graph: dict,
    model: str,
    fps: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
    frame_extractor: Callable = _extract_frames,
) -> Tuple[dict, dict]:
    if client is None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("PQSG OpenAI requires OPENAI_API_KEY")
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
    ids, mapping, _, step1, _ = _prompts(graph)
    frames = frame_extractor(video_path, fps=fps, jpeg_quality=70)
    if not frames:
        raise RuntimeError("PQSG extracted no video frames")
    parts = [{"type": "text", "text": step1}] + [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "low"},
        }
        for frame in frames
    ]
    messages = [{"role": "user", "content": parts}]
    r1 = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=16384)
    initial = (r1.choices[0].message.content or "").strip()
    if not initial:
        raise RuntimeError("PQSG OpenAI reasoning response was empty")
    followup = f"Based on your previous answers, give a yes/no for each question. Respond ONE per line: [Question ID]: yes OR [Question ID]: no. Answer all {len(ids)} questions."
    r2 = client.chat.completions.create(
        model=model,
        messages=[
            messages[0],
            {"role": "assistant", "content": initial},
            {"role": "user", "content": followup},
        ],
        max_completion_tokens=8192,
    )
    final = (r2.choices[0].message.content or "").strip()
    return _answers(ids, mapping, initial, final), {
        "requested_model": model,
        "returned_models": [_response_model(r1), _response_model(r2)],
        "request_ids": [_response_id(r1), _response_id(r2)],
        "raw_responses": [initial, final],
        "frame_count": len(frames),
        "coverage": _answer_coverage(final, ids),
    }


def _correct(answers: dict) -> Dict[str, bool]:
    return {
        qid: bool(data.get("is_correct"))
        for category in CATEGORIES
        for qid, data in answers.get(category, {}).items()
    }


def _parents_children(edges: list) -> Tuple[dict, dict]:
    parents, children = defaultdict(list), defaultdict(list)
    for edge in edges:
        for parent in edge["from"]:
            parents[edge["to"]].append(parent)
            children[parent].append(edge["to"])
    return parents, children


def tree_score(answers: dict, edges: list) -> float:
    correct = _correct(answers)
    _, children = _parents_children(edges)
    failed = {n for n, v in correct.items() if not v}
    propagated = set(failed)
    stack = list(failed)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in propagated:
                propagated.add(child)
                stack.append(child)
    return (len(correct) - len(propagated)) / len(correct)


def category_scores(answers: dict, edges: list) -> Dict[str, Optional[float]]:
    correct = _correct(answers)
    eligibility = eligibility_mask(answers, edges)

    out = {}
    for label, category in (
        ("object", "object_existence"),
        ("action", "action_verification"),
        ("physics", "physics"),
    ):
        nodes = [qid for qid in answers.get(category, {}) if eligibility[qid]]
        out[label] = sum(correct[qid] for qid in nodes) / len(nodes) if nodes else None
    return out


def eligibility_mask(answers: dict, edges: list) -> Dict[str, bool]:
    """Return whether every prerequisite ancestor passed for each question."""
    correct = _correct(answers)
    parents, _ = _parents_children(edges)

    def eligible(node: str, seen: Optional[set] = None) -> bool:
        seen = set() if seen is None else seen
        return all(
            parent not in seen and correct.get(parent, False) and eligible(parent, seen | {parent})
            for parent in parents.get(node, [])
        )

    return {node: eligible(node) for node in correct}
