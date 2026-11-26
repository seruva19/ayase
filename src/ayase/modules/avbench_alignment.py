"""AVBench audio-video matching probability from its published AV head."""

import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

MODEL_ID = "iiiiii123/AVBench_model"
MODEL_REVISION = "ea995265a1b5091bc06a4e8e281e64e3c5c0dc7e"
MODEL_FOLDER = "Qwen2.5-Omni-7B-AudioVideoMatching-Merged"


def _read_audio(path):
    import av

    from ayase.audio import extract_audio_with_ffmpeg
    from ayase.managed_ffmpeg import resolve_media_tool

    with av.open(str(path)) as media:
        if not media.streams.audio:
            return None
        sample_rate = media.streams.audio[0].rate
    if sample_rate and sample_rate > 48000:
        with TemporaryDirectory(prefix="ayase-avbench-") as work:
            intermediate = Path(work) / "audio.wav"
            result = subprocess.run(
                [
                    str(resolve_media_tool("ffmpeg")),
                    "-nostdin",
                    "-i",
                    str(path),
                    "-vn",
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(intermediate),
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                audio = extract_audio_with_ffmpeg(intermediate, target_sr=16000)
                if audio is not None:
                    return audio[:158400]
    audio = extract_audio_with_ffmpeg(path, target_sr=16000)
    return None if audio is None else audio[:158400]


class AVBenchAlignmentModule(PipelineModule):
    name = "avbench_alignment"
    description = "AVBench Audio-Video Matching head (higher = more consistent)"
    input_type = "video + audio"
    implementation_status = "implemented"
    provenance = "published"
    default_config = {"model_path": None, "device_map": "auto"}
    models = [
        {
            "id": MODEL_ID,
            "type": "huggingface",
            "url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}/{MODEL_FOLDER}",
            "task": "AVBench Qwen2.5-Omni-7B Audio-Video Matching head",
            "auto_download": True,
        }
    ]
    metric_groups = {"avbench_alignment_score": "alignment"}
    metric_info = {
        "avbench_alignment_score": "AVBench matched probability (0–1, higher means more consistent)"
    }
    sources = {"avbench_alignment_score": "AVBench — https://github.com/YaJialiang/AVBench"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._backend = None

    def setup(self):
        if self.test_mode:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

            from ayase.config import download_hf_snapshot

            model_path = self.config.get("model_path") or (
                download_hf_snapshot(
                    MODEL_ID,
                    self.config.get("models_dir", "models"),
                    revision=MODEL_REVISION,
                    allow_patterns=[f"{MODEL_FOLDER}/*"],
                )
                / MODEL_FOLDER
            )
            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map=self.config["device_map"],
                local_files_only=True,
            ).eval()
            self._processor = processor
            self._backend = "avbench_qwen2_5_omni"
        except Exception as exc:
            logger.warning("AVBench setup failed: %s", exc)

    def _score(self, path):
        import torch
        from qwen_vl_utils import process_vision_info

        audio = _read_audio(path)
        if audio is None or not audio.size:
            return None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(path), "fps": 1.0, "max_pixels": 480 * 640},
                    {"type": "audio", "audio": str(path)},
                    {"type": "text", "text": "Is this audio and video properly matched?"},
                ],
            }
        ]
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        processor = self._processor
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=images,
            videos=videos,
            audio=[audio],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        tokenizer = processor.tokenizer
        token_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("Yes", "No")]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return float(output.scores[0][0, token_ids].float().cpu().softmax(dim=0)[0])

    def process(self, sample: Sample) -> Sample:
        if self._model is None or not sample.is_video:
            return sample
        try:
            score = self._score(sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.avbench_alignment_score = score
        except Exception as exc:
            logger.warning("AVBench failed for %s: %s", sample.path, exc)
        return sample

    def teardown(self):
        self._model = self._processor = self._backend = None
