"""Owned adapter for the pinned MIT AudioJudge prompt and audio protocol.

Message ordering and homogeneous 24 kHz mono PCM16 concatenation follow the
released source. Heterogeneous WAV input is deliberately normalized to mono
signed PCM16 before concatenation, repairing its raw-byte reinterpretation bug.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .system_prompts import SYSTEM_PROMPTS as PAIRWISE_PROMPTS
from .system_prompts_pointwise import SYSTEM_PROMPTS as POINTWISE_PROMPTS

SOURCE_REVISION = "fc4a1f93fbe10a7e7fd8ea7298269d6e421a57ab"
PAIRWISE_METHODS = (
    "no_concatenation",
    "pair_example_concatenation",
    "examples_concatenation",
    "test_concatenation",
    "examples_and_test_concatenation",
)
POINTWISE_METHODS = ("no_concatenation", "examples_concatenation")

_ASSET_SHA256 = {
    "audio.wav": "8a2702b6c11f596937021e4940ef2c9217f997a2840dfe15d56380c8801844ed",
    "audio_1.wav": "02c1c055a3d00969b1cee862c1b10471ab6430be39825d045373c506d0f90f39",
    "audio_2.wav": "62a0f6be465aece2dd38284af5de5f293519eeadddc5b8dd55051c6301965c61",
    "audio_response.wav": "394ebdbc256f4cc8b187a1cb62c53d8b3e7ef4c781ebea6e0ce6004828e10423",
    "example_0.wav": "e19767a6deed5c756955b6fc73f83fe8edb7506c2dd86dca44ecd91c2e02acf9",
    "example_1.wav": "92b087749c99088474d60c4440cfc0330d1f503b38d035bf37c2b3e01d0594fa",
    "example_2.wav": "308a6b8a9c8c669cde91bf9d73a968cf61a7b624e5241cb357203680fa9a75b1",
    "example_3.wav": "f133b4e8cf03e2fb418a2a9e1a2b3c682ee747dc3fb5944ec109375650cbbe61",
    "example_4.wav": "28feea6b8fcb325a274d983f6fbb1ce46fb78d056b994cd20781c111d1bd90ad",
    "example_5.wav": "c03cc4fc46dec87144529a416d236e20267f514292de2105a29d378cdb430877",
    "example_6.wav": "9879d2c7b65fd81305450c8d7a2e5fc8ab76c316af4f3b79975a4e1cb768be91",
    "example_7.wav": "8b409468a70afbfdeeb8c1602b45e059bab2624bd2797caabc6f2a4b5b4117f4",
    "example_8.wav": "ea698453380a72c28f2e5a5611ecffd02a0f9c1b71dfb55ddb2333cf17fde7c7",
    "example_9.wav": "9050924f82f73ea554592fbe9ef2df28ba5fff4d1afeced75da89ff73352e380",
    "example_10.wav": "fc034b33b57edc253a0dc0b55a06bbb83173345e5cc620478dcf61970952fdc9",
    "first_audio.wav": "ffda48ad790f672ef48017b2bb87cd58d6a012a347440e0df88991822f095b35",
    "instruction.wav": "63473727d3e4f4f106538edbe7cd6df37eccd629640921e8bb2e0052c9c8be04",
    "second_audio.wav": "465b568bed61febcfec4a8ee9f684f5816837649d77c8ab6d4ce3bb589af8217",
    "test.wav": "1f0d4cc9ea09ba12708bf1b0ca2f36d5740415b4823cd60ccf9395e62f1bc102",
    "test_audio.wav": "26913ba9d3777b831e65f78e24a1cf0840125f3950357492ff59644473477e5d",
}


@dataclass(frozen=True)
class PairwiseExample:
    audio1_path: Path
    audio2_path: Path
    output: str
    instruction_path: Optional[Path] = None


@dataclass(frozen=True)
class PointwiseExample:
    audio_path: Path
    output: str


@dataclass(frozen=True)
class ResponseContract:
    mode: str
    task: str
    variant: str
    semantic: str
    result_key: str
    result_type: str
    allowed: Tuple[Any, ...]
    reasoning_key: Optional[str]
    system_prompt: str

    @property
    def exact_keys(self) -> frozenset[str]:
        keys = {self.result_key}
        if self.reasoning_key is not None:
            keys.add(self.reasoning_key)
        return frozenset(keys)


def canonical_contract(mode: str, task: str, variant: str) -> ResponseContract:
    """Resolve one of the 50 literal released prompt/schema contracts."""
    if mode == "pairwise":
        prompts = PAIRWISE_PROMPTS
        if task in {"pronunciation", "speaker"}:
            semantic, key, kind, allowed = "match", "match", "boolean", (False, True)
        elif task == "speed":
            semantic, key, kind, allowed = "choice", "label", "string", ("1", "2")
        else:
            ties = task in {"speakbench", "speakbench508", "chatbotarena", "chatbotarena_BA"}
            semantic, key, kind = "preference", "label", "string"
            allowed = ("1", "2", "tie") if ties else ("1", "2")
    elif mode == "pointwise":
        prompts = POINTWISE_PROMPTS
        semantic, key, kind, allowed = "score", "score", "integer", (1, 2, 3, 4, 5)
    else:
        raise ValueError("mode must be 'pairwise' or 'pointwise'")
    try:
        prompt = prompts[task][variant]
    except KeyError as exc:
        raise ValueError(f"unsupported AudioJudge task/variant: {mode}/{task}/{variant}") from exc
    reasoning = None if variant == "no_cot" else "reasoning"
    return ResponseContract(mode, task, variant, semantic, key, kind, allowed, reasoning, prompt)


def custom_contract(
    mode: str, prompt: str, semantic: str, schema: Mapping[str, Any]
) -> ResponseContract:
    """Build an explicit custom-prompt contract without inferring its meaning."""
    if mode not in {"pairwise", "pointwise"}:
        raise ValueError("mode must be 'pairwise' or 'pointwise'")
    if semantic not in {"score", "preference", "match", "choice"}:
        raise ValueError("custom semantic must be score, preference, match, or choice")
    required = {"result_key", "result_type", "allowed", "reasoning_key"}
    if set(schema) != required:
        raise ValueError(
            "custom schema must contain exactly result_key/result_type/allowed/reasoning_key"
        )
    key, kind = schema["result_key"], schema["result_type"]
    allowed, reasoning = schema["allowed"], schema["reasoning_key"]
    if not isinstance(key, str) or not key or kind not in {"boolean", "integer", "string"}:
        raise ValueError("custom schema has an invalid result key or type")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("custom schema allowed must be a nonempty list")
    if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
        raise ValueError("custom schema reasoning_key must be a nonempty string or null")
    if reasoning == key:
        raise ValueError("custom schema reasoning_key must differ from result_key")
    primitive = {"boolean": bool, "integer": int, "string": str}[kind]
    if any(type(item) is not primitive for item in allowed):
        raise ValueError("custom schema allowed values do not match result_type")
    expected = {
        "score": ("integer", (1, 2, 3, 4, 5)),
        "preference": ("string", ("1", "2", "tie")),
        "match": ("boolean", (False, True)),
        "choice": ("string", ("1", "2")),
    }
    if (kind, tuple(allowed)) != expected[semantic]:
        raise ValueError("custom schema does not match the declared semantic mapping")
    return ResponseContract(
        mode, "custom", "custom", semantic, key, kind, tuple(allowed), reasoning, prompt
    )


def parse_response(text: str, contract: ResponseContract) -> Tuple[float, Dict[str, Any]]:
    """Strictly parse a response and map it to the declared Ayase field value."""
    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"response contains duplicate key: {key}")
            value[key] = item
        return value

    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or frozenset(value) != contract.exact_keys:
        raise ValueError(f"response keys must be exactly {sorted(contract.exact_keys)}")
    if contract.reasoning_key is not None and type(value[contract.reasoning_key]) is not str:
        raise ValueError("response reasoning must be a JSON string")
    result = value[contract.result_key]
    expected_types = {"boolean": bool, "integer": int, "string": str}
    if type(result) is not expected_types[contract.result_type]:
        raise ValueError(f"response {contract.result_key} has the wrong JSON primitive type")
    if result not in contract.allowed:
        raise ValueError(f"response {contract.result_key} is outside the declared schema")
    if contract.semantic == "score":
        mapped = float(result)
    elif contract.semantic == "match":
        mapped = 1.0 if result else 0.0
    elif contract.semantic == "choice":
        mapped = float(result)
    else:
        mapped = {"1": 1.0, "tie": 0.0, "2": -1.0}[result]
    return mapped, value


def _asset(name: str) -> Path:
    path = Path(__file__).with_name("signal_audios") / name
    if not path.is_file():
        raise FileNotFoundError(f"required AudioJudge signal asset is missing: {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _ASSET_SHA256[name]:
        raise ValueError(f"AudioJudge signal asset digest mismatch: {name}")
    return path


def _wav_info(path: Path) -> Tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as stream:
        if stream.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {path}")
        return (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.readframes(stream.getnframes()),
        )


def _pcm16_mono(path: Path, target_rate: int) -> bytes:
    import audioop

    rate, channels, width, frames = _wav_info(path)
    if width == 1:
        frames = audioop.bias(frames, 1, -128)
    if channels == 2:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(f"WAV must have one or two channels: {path}")
    if width != 2:
        frames = audioop.lin2lin(frames, width, 2)
    if rate != target_rate:
        frames, _ = audioop.ratecv(frames, 2, 1, rate, target_rate, None)
    return frames


def _write_wav(path: Path, rate: int, frames: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(frames)
    return path


def normalize_wav(source: Path, output: Path) -> Path:
    """Write a full-duration mono PCM16 WAV, retaining the input sample rate."""
    rate, channels, width, _ = _wav_info(source)
    if channels == 1 and width == 2:
        return source
    return _write_wav(output, rate, _pcm16_mono(source, rate))


def _modal_rate(paths: Sequence[Path]) -> int:
    rates = [_wav_info(path)[0] for path in paths]
    return Counter(rates).most_common(1)[0][0] if rates else 24000


def concatenate_pair_audio(
    groups: Sequence[Sequence[Path]],
    output: Path,
    *,
    is_test: bool,
    start_index: int = 1,
) -> Path:
    """Concatenate pair or instruction/pair groups using released boundary order."""
    if not groups or any(len(group) not in {2, 3} for group in groups):
        raise ValueError("pair groups must contain audio1/audio2 or instruction/audio1/audio2")
    instruction = len(groups[0]) == 3
    if any((len(group) == 3) != instruction for group in groups):
        raise ValueError("all pair groups must use the same instruction mode")
    flat = [Path(path) for group in groups for path in group]
    rate = _modal_rate(flat)
    gap = bytes(int(0.5 * rate) * 2)
    chunks: List[bytes] = []
    for offset, group in enumerate(groups):
        label = "test.wav" if is_test else f"example_{start_index + offset}.wav"
        chunks.extend((_pcm16_mono(_asset(label), rate), gap))
        if instruction:
            chunks.extend((_pcm16_mono(_asset("instruction.wav"), rate), gap))
            chunks.extend((_pcm16_mono(Path(group[0]), rate), gap))
            audio1, audio2 = Path(group[1]), Path(group[2])
        else:
            audio1, audio2 = Path(group[0]), Path(group[1])
        chunks.extend((_pcm16_mono(_asset("audio_1.wav"), rate), gap))
        chunks.extend((_pcm16_mono(audio1, rate), gap))
        chunks.extend((_pcm16_mono(_asset("audio_2.wav"), rate), gap))
        chunks.extend((_pcm16_mono(audio2, rate), gap))
    return _write_wav(output, rate, b"".join(chunks))


def concatenate_pointwise_audio(paths: Sequence[Path], output: Path) -> Path:
    """Concatenate pointwise examples with the released 0.7-second trailing gap."""
    if not paths:
        raise ValueError("at least one pointwise example is required")
    rate = _modal_rate([Path(path) for path in paths])
    gap = bytes(int(0.5 * rate) * 2)
    tail = bytes(int(0.7 * rate) * 2)
    chunks: List[bytes] = []
    for index, path in enumerate(paths, 1):
        chunks.extend((_pcm16_mono(_asset(f"example_{index}.wav"), rate), gap))
        chunks.extend((_pcm16_mono(_asset("audio.wav"), rate), gap))
        chunks.extend((_pcm16_mono(Path(path), rate), tail))
    return _write_wav(output, rate, b"".join(chunks))


def _openai_audio(path: Path) -> Dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


def _gemini_audio(path: Path) -> Dict[str, Any]:
    rate = 16000
    frames = _pcm16_mono(path, rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(frames)
    return {"mime_type": "audio/wav", "data": buffer.getvalue()}


def _example_text(outputs: Sequence[str]) -> str:
    text = "Examples information:\n"
    for index, output in enumerate(outputs, 1):
        text += f"Example {index}:\n- Expected output: {output}\n\n"
    return text


def _pair_paths(example: PairwiseExample, instruction: bool) -> List[Path]:
    if instruction:
        if example.instruction_path is None:
            raise ValueError("every instruction-mode example needs instruction_path")
        return [example.instruction_path, example.audio1_path, example.audio2_path]
    return [example.audio1_path, example.audio2_path]


def _pointwise_openai_messages(
    candidate_path: Path,
    system_prompt: str,
    work_dir: Path,
    examples: Sequence[PointwiseExample],
    user_prompt: Optional[str],
    concatenation_method: str,
) -> List[Dict[str, Any]]:
    if concatenation_method not in POINTWISE_METHODS:
        raise ValueError("unsupported pointwise concatenation method")
    prompt = (
        "Please provide your response according to this audio clip:"
        if user_prompt is None
        else user_prompt
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if concatenation_method == "no_concatenation" and examples:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "Here are some examples for reference:"}],
            }
        )
        for index, example in enumerate(examples, 1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Example {index}:"},
                        _openai_audio(example.audio_path),
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": example.output})
        messages.append(
            {
                "role": "assistant",
                "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.",
            }
        )
    elif examples:
        joined = concatenate_pointwise_audio(
            [example.audio_path for example in examples], work_dir / "pointwise_examples.wav"
        )
        content = [
            {"type": "text", "text": "Here are some examples for reference:"},
            _openai_audio(joined),
            {"type": "text", "text": _example_text([example.output for example in examples])},
        ]
        messages.append({"role": "user", "content": content})
        messages.append(
            {
                "role": "assistant",
                "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.",
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this audio clip:"},
                _openai_audio(candidate_path),
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def _separate_pair_content(
    paths: Sequence[Path], prompt: str, *, instruction: bool, example: bool
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if instruction:
        target = "example" if example else "test"
        content.extend(
            (
                {"type": "text", "text": f"Here is the instruction for this {target}:"},
                _openai_audio(paths[0]),
            )
        )
        paths = paths[1:]
    content.extend(
        (
            {"type": "text", "text": "Here is the first audio clip:"},
            _openai_audio(paths[0]),
            {"type": "text", "text": "Here is the second audio clip:"},
            _openai_audio(paths[1]),
            {"type": "text", "text": prompt},
        )
    )
    return content


def build_openai_messages(
    *,
    mode: str,
    candidate_path: Path,
    system_prompt: str,
    work_dir: Path,
    comparator_path: Optional[Path] = None,
    instruction_path: Optional[Path] = None,
    examples: Sequence[Any] = (),
    user_prompt: Optional[str] = None,
    concatenation_method: str,
) -> List[Dict[str, Any]]:
    """Build source-conformant OpenAI chat messages without service calls."""
    if mode == "pointwise":
        return _pointwise_openai_messages(
            candidate_path,
            system_prompt,
            work_dir,
            examples,
            user_prompt,
            concatenation_method,
        )
    if mode != "pairwise" or comparator_path is None:
        raise ValueError("pairwise mode requires comparator_path")
    if concatenation_method not in PAIRWISE_METHODS:
        raise ValueError("unsupported pairwise concatenation method")
    prompt = (
        "Please provide your response according to these audio clips:"
        if user_prompt is None
        else user_prompt
    )
    pair_examples: List[PairwiseExample] = list(examples)
    instruction = instruction_path is not None
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if concatenation_method in {"no_concatenation", "test_concatenation"}:
        for example in pair_examples:
            paths = _pair_paths(example, instruction)
            messages.append(
                {
                    "role": "user",
                    "content": _separate_pair_content(
                        paths, prompt, instruction=instruction, example=True
                    ),
                }
            )
            messages.append({"role": "assistant", "content": example.output})
    elif concatenation_method == "pair_example_concatenation":
        for index, example in enumerate(pair_examples, 1):
            joined = concatenate_pair_audio(
                [_pair_paths(example, instruction)],
                work_dir / f"example_pair_{index}.wav",
                is_test=False,
                start_index=index,
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze these audio clips:"},
                        _openai_audio(joined),
                    ],
                }
            )
            messages.append({"role": "assistant", "content": example.output})
    elif pair_examples:
        joined = concatenate_pair_audio(
            [_pair_paths(example, instruction) for example in pair_examples],
            work_dir / "all_examples.wav",
            is_test=False,
        )
        content = [
            {"type": "text", "text": "Here are some examples for reference:"},
            _openai_audio(joined),
            {"type": "text", "text": _example_text([example.output for example in pair_examples])},
        ]
        messages.append({"role": "user", "content": content})
        messages.append(
            {
                "role": "assistant",
                "content": "I understand these examples. I'll apply this understanding to analyze the new audio clips you provide.",
            }
        )

    test_paths: List[Path] = (
        [instruction_path, candidate_path, comparator_path]
        if instruction_path is not None
        else [candidate_path, comparator_path]
    )
    if concatenation_method in {"test_concatenation", "examples_and_test_concatenation"}:
        joined = concatenate_pair_audio([test_paths], work_dir / "test.wav", is_test=True)
        content = [{"type": "text", "text": prompt}, _openai_audio(joined)]
    else:
        content = _separate_pair_content(
            test_paths, prompt, instruction=instruction, example=False
        )
    messages.append({"role": "user", "content": content})
    return messages


def _pointwise_gemini_messages(
    candidate_path: Path,
    system_prompt: str,
    work_dir: Path,
    examples: Sequence[PointwiseExample],
    user_prompt: Optional[str],
    concatenation_method: str,
) -> List[Any]:
    if concatenation_method not in POINTWISE_METHODS:
        raise ValueError("unsupported pointwise concatenation method")
    prompt = (
        "Please provide your response according to this audio clip:"
        if user_prompt is None
        else user_prompt
    )
    messages: List[Any] = [system_prompt]
    if concatenation_method == "no_concatenation" and examples:
        messages.append("Here are some examples for reference:")
        for index, example in enumerate(examples, 1):
            messages.extend(
                (
                    f"Example {index}:",
                    _gemini_audio(example.audio_path),
                    prompt,
                    "Here is the assistant's response for this example:",
                    example.output,
                )
            )
        messages.append(
            "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide."
        )
    elif examples:
        joined = concatenate_pointwise_audio(
            [example.audio_path for example in examples], work_dir / "pointwise_examples.wav"
        )
        messages.extend(
            (
                "Here are some examples for reference:",
                _gemini_audio(joined),
                _example_text([example.output for example in examples]),
                "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.",
            )
        )
    messages.extend(("Please analyze this audio clip:", _gemini_audio(candidate_path), prompt))
    return messages


def _gemini_separate_pair(
    messages: List[Any], paths: Sequence[Path], *, instruction: bool, example: bool
) -> None:
    if instruction:
        target = "example" if example else "test"
        messages.extend((f"Here is the instruction for this {target}:", _gemini_audio(paths[0])))
        paths = paths[1:]
    messages.extend(
        (
            "Here is the first audio clip:",
            _gemini_audio(paths[0]),
            "Here is the second audio clip:",
            _gemini_audio(paths[1]),
        )
    )


def build_gemini_messages(
    *,
    mode: str,
    candidate_path: Path,
    system_prompt: str,
    work_dir: Path,
    comparator_path: Optional[Path] = None,
    instruction_path: Optional[Path] = None,
    examples: Sequence[Any] = (),
    user_prompt: Optional[str] = None,
    concatenation_method: str,
) -> List[Any]:
    """Build the released flat Gemini content sequence without service calls."""
    if mode == "pointwise":
        return _pointwise_gemini_messages(
            candidate_path,
            system_prompt,
            work_dir,
            examples,
            user_prompt,
            concatenation_method,
        )
    if mode != "pairwise" or comparator_path is None:
        raise ValueError("pairwise mode requires comparator_path")
    if concatenation_method not in PAIRWISE_METHODS:
        raise ValueError("unsupported pairwise concatenation method")
    if user_prompt is None:
        prompt = (
            "Please analyze these audio clips:"
            if instruction_path is not None
            else "Please provide your response according to these audio clips:"
        )
    else:
        prompt = user_prompt
    pair_examples: List[PairwiseExample] = list(examples)
    instruction = instruction_path is not None
    messages: List[Any] = [system_prompt]

    if concatenation_method in {"no_concatenation", "test_concatenation"}:
        for example in pair_examples:
            _gemini_separate_pair(
                messages, _pair_paths(example, instruction), instruction=instruction, example=True
            )
            messages.extend(
                (
                    "Please analyze these audio clips:",
                    "Here is the assistant's response for this example:",
                    example.output,
                )
            )
    elif concatenation_method == "pair_example_concatenation":
        for index, example in enumerate(pair_examples, 1):
            joined = concatenate_pair_audio(
                [_pair_paths(example, instruction)],
                work_dir / f"example_pair_{index}.wav",
                is_test=False,
                start_index=index,
            )
            messages.extend(
                (
                    "Please analyze these audio clips:",
                    _gemini_audio(joined),
                    "Here is the assistant's response for this example:",
                    example.output,
                )
            )
    elif pair_examples:
        joined = concatenate_pair_audio(
            [_pair_paths(example, instruction) for example in pair_examples],
            work_dir / "all_examples.wav",
            is_test=False,
        )
        messages.extend(
            (
                "Here are some examples for reference:",
                _gemini_audio(joined),
                _example_text([example.output for example in pair_examples]),
            )
        )

    test_paths: List[Path] = (
        [instruction_path, candidate_path, comparator_path]
        if instruction_path is not None
        else [candidate_path, comparator_path]
    )
    if concatenation_method in {"test_concatenation", "examples_and_test_concatenation"}:
        joined = concatenate_pair_audio([test_paths], work_dir / "test.wav", is_test=True)
        if instruction:
            messages.append("Please provide your response according to these audio clips:")
        else:
            messages.append("Please analyze these audio clips:")
        messages.extend((_gemini_audio(joined), prompt))
    else:
        _gemini_separate_pair(
            messages, test_paths, instruction=instruction, example=False
        )
        messages.append(prompt)
    return messages
