"""Shared helpers for prompt-conditioned image reward modules."""

from __future__ import annotations

import base64
import json
import logging
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ayase.models import Sample

logger = logging.getLogger(__name__)


def get_prompt(sample: Sample, config: Dict[str, Any], key: str = "prompt") -> Optional[str]:
    override = config.get(key)
    if override:
        return str(override).strip()

    if sample.caption and sample.caption.text.strip():
        return sample.caption.text.strip()

    txt_path = sample.path.with_suffix(".txt")
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:
        logger.debug("Failed to read prompt sidecar: %s", txt_path)
        return None


def load_rgb_image(
    path: Path,
    max_image_size: int = 0,
    resize_to_square: bool = False,
) -> Optional[Image.Image]:
    try:
        image = Image.open(path).convert("RGB")
        if max_image_size > 0 and max(image.size) > max_image_size:
            if resize_to_square:
                image = image.resize((max_image_size, max_image_size), Image.LANCZOS)
            else:
                scale = max_image_size / max(image.size)
                size = (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                )
                image = image.resize(size, Image.LANCZOS)
        image.load()
        return image
    except Exception as e:
        logger.debug("Failed to load image %s: %s", path, e)
        return None


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat_completions_url(endpoint_url: str) -> str:
    endpoint = str(endpoint_url).rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def openai_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_openai_chat(
    endpoint_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: float = 600.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        chat_completions_url(endpoint_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=openai_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
