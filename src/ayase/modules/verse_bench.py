"""Ayase-native Verse-Bench dataset-level benchmark runner.

Uses the vendored Verse-Bench inferencers directly and stores the aggregate
outputs in ``DatasetStats``.

Benchmark assets are not bundled. To execute this module, provide a
materialized Verse-Bench dataset root via ``dataset_root`` or
``VERSE_BENCH_DATASET_ROOT``.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


_METRIC_BOUNDS = {
    "AS": {"best": 1.0, "worst": 0.0},
    "ID": {"best": 1.0, "worst": 0.0},
    "FD": {"best": 0.0, "worst": 3.0},
    "KL": {"best": 0.0, "worst": 4.0},
    "CS": {"best": 1.0, "worst": 0.0},
    "CE": {"best": 10.0, "worst": 1.0},
    "CU": {"best": 10.0, "worst": 1.0},
    "PC": {"best": 1.0, "worst": 10.0},
    "PQ": {"best": 10.0, "worst": 1.0},
    "WER": {"best": 0.0, "worst": 1.0},
    "LSE-C": {"best": 10.0, "worst": 0.0},
    "AV-A": {"best": 0.0, "worst": 1.0},
}

_HIGHER_IS_BETTER_METRICS = {"AS", "ID", "CS", "CE", "CU", "PQ", "LSE-C"}

_WEIGHTS = {
    "joint": 0.5,
    "video": 0.2,
    "audio": 0.2,
    "other": 0.1,
}


class VerseBenchModule(PipelineModule):
    name = "verse_bench"
    description = "Ayase-native Verse-Bench dataset-level benchmark wrapper"
    default_config = {
        "dataset_root": None,
        "input_dir": None,
        "models_path": "models",
    }
    models = [
        {"id": "google/siglip-so400m-patch14-384", "type": "huggingface", "task": "SigLIP vision encoder for Aesthetic Predictor V2.5"},
        {"id": "aesthetic_predictor_v2_5.pth", "type": "local", "task": "Aesthetic Predictor V2.5 head weights"},
        {"id": "ckpt_koniq10k.pt", "type": "local", "task": "MANIQA Swin-T quality assessment (KonIQ-10k)"},
        {"id": "audiobox-aesthetics", "type": "pip_package", "install": "pip install audiobox_aesthetics", "task": "Meta AudioBox audio quality (CE, CU, PC, PQ)"},
        {"id": "facebook/dinov2-large", "type": "huggingface", "task": "DINOv2 ViT-L/16 image identity features"},
        {"id": "630k-audioset-fusion-best.pt", "type": "local", "task": "LAION CLAP fusion model for audio-text similarity and FAD"},
        {"id": "roberta-base", "type": "huggingface", "task": "RoBERTa text encoder (CLAP submodule)"},
        {"id": "hear21passt", "type": "pip_package", "install": "pip install hear21passt", "task": "PaSST audio model for KL divergence"},
        {"id": "24-01-04T16-39-21.pt", "type": "local", "task": "Syncformer AV sync model (AST + MotionFormer)"},
        {"id": "syncnet_v2.model", "type": "local", "task": "SyncNet v2 lip-sync model"},
        {"id": "FunAudioLLM/SenseVoiceSmall", "type": "huggingface", "task": "SenseVoice ASR for WER computation"},
        {"id": "fsmn-vad", "type": "huggingface", "task": "FSMN voice activity detection (FunASR)"},
    ]
    metric_info = {
        "verse_bench_overall": "Weighted aggregate score (0-1, higher=better) from S_joint(50%), S_video(20%), S_audio(20%), S_other(10%)",
        "verse_bench_metrics": "Raw 12-component metric dict: AS, ID, FD, KL, CS, CE, CU, PC, PQ, WER, LSE-C, AV-A",
        "verse_bench_breakdown": "Subscore dict: S_joint, S_video, S_audio, S_other, Overall Score",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = None
        self._dataset_root: Optional[Path] = None
        self._vendor_root: Optional[Path] = None

    def setup(self) -> None:
        vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verse_bench"
        if not vendor_root.exists():
            logger.warning("Verse-Bench vendor root not found: %s", vendor_root)
            return

        self._vendor_root = vendor_root
        self._backend = "native"

    def process(self, sample: Sample) -> Sample:
        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        if self._backend != "native":
            return

        try:
            input_dir = self._resolve_input_dir(all_samples)
            if input_dir is None:
                return

            dataset_root = self._dataset_root or self._resolve_dataset_root()
            if dataset_root is None:
                return
            if not self._validate_benchmark_assets(dataset_root):
                return

            raw_metrics, breakdown = self._run_native_benchmark(input_dir, dataset_root)
            if raw_metrics is None or breakdown is None:
                return

            if hasattr(self, "pipeline") and self.pipeline and hasattr(self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric("verse_bench_metrics", raw_metrics)
                self.pipeline.add_dataset_metric("verse_bench_breakdown", breakdown)
                overall = breakdown.get("Overall Score")
                if overall is not None:
                    self.pipeline.add_dataset_metric("verse_bench_overall", float(overall))
        except Exception as e:
            logger.warning("Verse-Bench post_process failed: %s", e)

    def _resolve_input_dir(self, all_samples: List[Sample]) -> Optional[Path]:
        explicit = self.config.get("input_dir")
        if explicit:
            path = Path(explicit)
            if path.exists():
                return path
            logger.warning("Verse-Bench input_dir does not exist: %s", path)
            return None

        video_samples = [sample for sample in all_samples if sample.is_video]
        if not video_samples:
            logger.warning("Verse-Bench requires video samples to infer input_dir.")
            return None

        parent_dirs = {sample.path.parent.resolve() for sample in video_samples}
        if len(parent_dirs) != 1:
            logger.warning(
                "Verse-Bench could not infer a single input_dir from %d video directories. "
                "Set module config 'input_dir' explicitly.",
                len(parent_dirs),
            )
            return None

        return next(iter(parent_dirs))

    def _resolve_dataset_root(self) -> Optional[Path]:
        explicit = self.config.get("dataset_root") or os.environ.get("VERSE_BENCH_DATASET_ROOT")
        if not explicit:
            logger.warning(
                "Verse-Bench dataset root not configured. Set 'dataset_root' or VERSE_BENCH_DATASET_ROOT."
            )
            return None
        path = Path(explicit)
        if not path.exists():
            logger.warning("Verse-Bench dataset root not found: %s", path)
            return None
        return path

    def _validate_benchmark_assets(self, dataset_root: Path) -> bool:
        set1_dir = dataset_root / "set1"
        set2_dir = dataset_root / "set2"
        set3_dir = dataset_root / "set3"

        if not set1_dir.exists():
            logger.warning("Verse-Bench dataset missing set1: %s", set1_dir)
            return False

        if len(list(set1_dir.glob("*.json"))) == 0 or len(list(set1_dir.glob("*.jpg"))) == 0:
            logger.warning("Verse-Bench set1 is incomplete: expected vendored .json and .jpg files.")
            return False

        set2_materialized = any(set2_dir.glob("*.wav")) and any(set2_dir.glob("*.jpg"))
        set3_materialized = any(set3_dir.glob("*.wav")) and any(set3_dir.glob("*.jpg"))
        if not set2_materialized or not set3_materialized:
            logger.warning(
                "Verse-Bench dataset is not fully materialized. "
                "The official scorer requires set2/set3 clip assets (.wav/.jpg) in addition to metadata."
            )
            return False

        return True

    def _run_native_benchmark(
        self, input_dir: Path, dataset_root: Path
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        inferencers = self._load_inferencers()
        if inferencers is None:
            return None, None

        totals = {
            "aesthetic": [],
            "musiq": [],
            "maniqa": [],
            "fd": [],
            "kl": [],
            "ce": [],
            "cu": [],
            "pc": [],
            "pq": [],
            "cs": [],
            "wer": [],
            "ava": [],
            "lse_c": [],
            "id": [],
        }

        self._evaluate_set("set1", input_dir, dataset_root, inferencers, totals)
        self._evaluate_set("set2", input_dir, dataset_root, inferencers, totals)
        self._evaluate_set("set3", input_dir, dataset_root, inferencers, totals)

        avg_aesthetic = self._mean_or_missing(totals["aesthetic"])
        avg_musiq = self._mean_or_missing(totals["musiq"])
        avg_maniqa = self._mean_or_missing(totals["maniqa"])
        avg_as = (
            (avg_aesthetic + avg_musiq + avg_maniqa) / 3.0
            if avg_aesthetic != -999 and avg_musiq != -999 and avg_maniqa != -999
            else -999
        )

        scores_dict = {
            "AS": avg_as,
            "ID": self._mean_or_missing(totals["id"]),
            "FD": self._mean_or_missing(totals["fd"]),
            "KL": self._mean_or_missing(totals["kl"]),
            "CS": self._mean_or_missing(totals["cs"]),
            "CE": self._mean_or_missing(totals["ce"]),
            "CU": self._mean_or_missing(totals["cu"]),
            "PC": self._mean_or_missing(totals["pc"]),
            "PQ": self._mean_or_missing(totals["pq"]),
            "WER": self._mean_or_missing(totals["wer"]),
            "LSE-C": self._mean_or_missing(totals["lse_c"]),
            "AV-A": self._mean_or_missing(totals["ava"]),
        }
        return scores_dict, self._calculate_overall_score(scores_dict)

    def _load_inferencers(self) -> Optional[Dict[str, object]]:
        if self._vendor_root is None:
            return None

        vendor_root_str = str(self._vendor_root)
        if vendor_root_str not in sys.path:
            sys.path.insert(0, vendor_root_str)

        try:
            from aesthetic.aesthetic_inferencer import AestheticInferencer
            from aesthetic.maniqa_inferencer import ManiqaInferencer
            from aesthetic.musiq_inferencer import MusiqInferencer
            from audio_box.audio_box_inferencer import AudioBoxInferencer
            from dino.dinov3_inferencer import DinoV3Inferencer
            from fd.clap_inferencer import ClapInferencer
            from kl.kld_inferencer import KLDInferencer
            from syncformer.syncformer_inferencer import SyncformerInferencer
            from syncnet.syncnet_inferencer import SyncnetInferencer
            from wer.wer_inferencer import WERInferencer
        except Exception as e:
            logger.warning("Verse-Bench inferencer imports failed: %s", e)
            return None

        models_path = str(self.config.get("models_path", "models"))
        os.environ["MODELS_PATH"] = models_path

        try:
            return {
                "aesthetic": AestheticInferencer(models_path),
                "musiq": MusiqInferencer(),
                "maniqa": ManiqaInferencer(models_path),
                "audio_box": AudioBoxInferencer(models_path),
                "syncformer": SyncformerInferencer(models_path),
                "syncnet": SyncnetInferencer(models_path),
                "wer": WERInferencer(models_path),
                "clap": ClapInferencer(models_path),
                "kld": KLDInferencer(),
                "dino": DinoV3Inferencer(models_path),
            }
        except Exception as e:
            logger.warning("Verse-Bench inferencer setup failed: %s", e)
            return None

    def _evaluate_set(
        self,
        set_name: str,
        input_dir: Path,
        dataset_root: Path,
        inferencers: Dict[str, object],
        totals: Dict[str, List[float]],
    ) -> None:
        set_dir = dataset_root / set_name
        for json_path in sorted(set_dir.glob("*.json")):
            base_name = json_path.stem
            try:
                item = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Verse-Bench failed to read %s: %s", json_path, e)
                continue

            video_path = input_dir / f"{base_name}.mp4"
            wav_path = input_dir / f"{base_name}.wav"
            image_path = set_dir / f"{base_name}.jpg"
            ref_wav_path = set_dir / f"{base_name}.wav"

            audio_prompt = self._get_audio_prompt(item)
            speech_text = self._get_speech_text(item)

            if video_path.exists():
                avg_aesthetic, avg_musiq, avg_maniqa = self._evaluate_aesthetic_video(
                    video_path,
                    inferencers["aesthetic"],
                    inferencers["musiq"],
                    inferencers["maniqa"],
                )
                totals["aesthetic"].append(avg_aesthetic)
                totals["musiq"].append(avg_musiq)
                totals["maniqa"].append(avg_maniqa)

                if audio_prompt and set_name in {"set1", "set2"}:
                    totals["ava"].append(inferencers["syncformer"].infer(str(video_path)))
                if speech_text:
                    sync_score = inferencers["syncnet"].infer(str(video_path))[1]
                    if sync_score is not None:
                        totals["lse_c"].append(sync_score)
                if image_path.exists():
                    totals["id"].append(
                        self._evaluate_dinov3_video(video_path, inferencers["dino"], image_path)
                    )

            if wav_path.exists():
                if set_name in {"set1", "set2"}:
                    ce, cu, pc, pq = self._evaluate_audiobox_wav(wav_path, inferencers["audio_box"])
                    totals["ce"].append(ce)
                    totals["cu"].append(cu)
                    totals["pc"].append(pc)
                    totals["pq"].append(pq)

                if audio_prompt and set_name in {"set1", "set2"}:
                    totals["cs"].append(inferencers["clap"].infer(str(wav_path), audio_prompt))
                    if set_name == "set2" and ref_wav_path.exists():
                        totals["fd"].append(
                            inferencers["clap"].infer_fd(str(wav_path), str(ref_wav_path))
                        )
                        totals["kl"].append(
                            inferencers["kld"].infer(str(wav_path), str(ref_wav_path))
                        )

                if speech_text:
                    totals["wer"].append(inferencers["wer"].infer_audio_text(str(wav_path), speech_text))

    def _evaluate_aesthetic_video(
        self,
        video_path: Path,
        aesthetic_inferencer,
        musiq_inferencer,
        maniqa_inferencer,
    ) -> Tuple[float, float, float]:
        from moviepy.editor import VideoFileClip
        from PIL import Image

        clip = VideoFileClip(str(video_path))
        try:
            aesthetic_scores = []
            musiq_scores = []
            maniqa_scores = []
            for frame in clip.iter_frames():
                image = Image.fromarray(frame)
                aesthetic_scores.append(aesthetic_inferencer.infer(image))
                musiq_scores.append(musiq_inferencer.infer(image))
                maniqa_scores.append(maniqa_inferencer.infer(image))
        finally:
            clip.close()

        avg_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores)
        avg_musiq = sum(musiq_scores) / len(musiq_scores)
        avg_maniqa = sum(maniqa_scores) / len(maniqa_scores)
        return avg_aesthetic / 10.0, avg_musiq / 100.0, avg_maniqa

    def _evaluate_audiobox_wav(self, wav_path: Path, audio_box_inferencer) -> Tuple[float, float, float, float]:
        score = audio_box_inferencer.infer(str(wav_path))
        return score["CE"], score["CU"], score["PC"], score["PQ"]

    def _evaluate_dinov3_video(self, video_path: Path, dinov3_inferencer, image_path: Path) -> float:
        from moviepy.editor import VideoFileClip
        from PIL import Image

        anchor_feature = dinov3_inferencer.get_feature(Image.open(image_path))
        clip = VideoFileClip(str(video_path))
        try:
            frame_cos = []
            for frame in clip.iter_frames():
                feature = dinov3_inferencer.get_feature(Image.fromarray(frame))
                frame_cos.append(dinov3_inferencer.infer_feature(feature, anchor_feature))
        finally:
            clip.close()
        return sum(frame_cos) / len(frame_cos)

    def _get_audio_prompt(self, item: dict) -> Optional[str]:
        prompt = item.get("audio_prompt")
        if isinstance(prompt, list):
            return prompt[0] if prompt else None
        if isinstance(prompt, str) and prompt:
            return prompt
        return None

    def _get_speech_text(self, item: dict) -> Optional[str]:
        speech_prompt = item.get("speech_prompt")
        if isinstance(speech_prompt, dict):
            text = speech_prompt.get("text")
            if isinstance(text, str) and text:
                return text
        return None

    def _mean_or_missing(self, values: List[float]) -> float:
        return (sum(values) / len(values)) if values else -999

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        bounds = _METRIC_BOUNDS[metric_name]
        best, worst = bounds["best"], bounds["worst"]

        if best == worst:
            return 0.5

        if metric_name in _HIGHER_IS_BETTER_METRICS:
            norm_score = (value - worst) / (best - worst)
        else:
            norm_score = (worst - value) / (worst - best)

        return max(0.0, min(1.0, norm_score))

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> Dict[str, float]:
        normalized_scores = {
            name: self._normalize_metric(name, value) for name, value in metrics.items()
        }

        s_video = (normalized_scores["AS"] + normalized_scores["ID"]) / 2

        audio_metrics = ["FD", "KL", "CS", "CE", "CU", "PC", "PQ"]
        s_audio = sum(normalized_scores[m] for m in audio_metrics) / len(audio_metrics)

        s_other = (normalized_scores["WER"] + normalized_scores["LSE-C"]) / 2

        cs_norm = normalized_scores["CS"]
        av_a_norm = normalized_scores["AV-A"]
        if (cs_norm + av_a_norm) == 0:
            s_joint = 0.0
        else:
            s_joint = 2 * (cs_norm * av_a_norm) / (cs_norm + av_a_norm)

        overall_score = (
            _WEIGHTS["joint"] * s_joint
            + _WEIGHTS["video"] * s_video
            + _WEIGHTS["audio"] * s_audio
            + _WEIGHTS["other"] * s_other
        )

        return {
            "S_joint": s_joint,
            "S_video": s_video,
            "S_audio": s_audio,
            "S_other": s_other,
            "Overall Score": overall_score,
        }
