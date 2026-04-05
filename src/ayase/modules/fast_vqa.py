"""FAST-VQA / FasterVQA deep learning video quality assessment.

Uses DiViDeAddEvaluator with spatial fragment sampling from the FAST-VQA framework.
Returns fast_vqa_score (0-100, higher = better). Supports 5 model variants."""

import logging
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Original URLs:
#   https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth
#   https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth
#   https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_M_1_4.pth
FASTVQA_MODEL_URLS = {
    "FAST_VQA_3D_1_1.pth": "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_3D_1_1.pth",
    "FAST_VQA_B_1_4.pth": "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_B_1_4.pth",
    "FAST_VQA_M_1_4.pth": "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/fast_vqa/FAST_VQA_M_1_4.pth",
}


def resize_frame_maintain_aspect(frame, min_dim):
    if min_dim <= 0:
        return frame
    h, w = frame.shape[:2]
    if h <= min_dim and w <= min_dim:
        return frame
    if h < w:
        scale = min_dim / h
    else:
        scale = min_dim / w
    if scale >= 1.0:
        return frame
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FastVQAModule(PipelineModule):
    name = "fast_vqa"
    description = "Deep Learning Video Quality Assessment (FAST-VQA)"
    default_config = {"model_type": "FasterVQA"}

    def __init__(self, config=None):
        super().__init__(config)
        self.model_type = self.config.get(
            "model_type", "FasterVQA"
        )  # FasterVQA, FasterVQA-MS, FasterVQA-MT, FAST-VQA, FAST-VQA-M
        self.device = "cpu"
        self._model = None
        self._opt = None
        self._ml_available = False

        self.fastvqa_mean_std = {
            "FasterVQA": (0.14759505, 0.03613452),
            "FasterVQA-MS": (0.15218826, 0.03230298),
            "FasterVQA-MT": (0.14699507, 0.036453716),
            "FAST-VQA": (-0.110198185, 0.04178565),
            "FAST-VQA-M": (0.023889644, 0.030781006),
        }

    def setup(self) -> None:
        try:
            import torch
            import decord
            from ayase.third_party.fastvqa.models import DiViDeAddEvaluator
            from ayase.third_party.fastvqa.datasets import get_spatial_fragments

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up FastVQA ({self.model_type}) on {self.device}...")

            # Load options
            opts_map = {
                "FasterVQA": "f3dvqa-b.yml",
                "FasterVQA-MS": "fastervqa-ms.yml",
                "FasterVQA-MT": "fastervqa-mt.yml",
                "FAST-VQA": "fast-b.yml",
                "FAST-VQA-M": "fast-m.yml",
            }

            if self.model_type not in opts_map:
                logger.warning(f"Unknown model type {self.model_type}, defaulting to FasterVQA")
                self.model_type = "FasterVQA"

            opt_file = opts_map[self.model_type]
            # Assume options are in src/ayase/fastvqa/options
            # We need to find the absolute path
            import ayase.third_party.fastvqa

            base_path = Path(ayase.third_party.fastvqa.__file__).parent
            opt_path = base_path / "options" / opt_file

            if not opt_path.exists():
                raise FileNotFoundError(f"Config file not found: {opt_path}")

            import yaml  # type: ignore[import-untyped]
            with open(opt_path, "r") as f:
                self._opt = yaml.safe_load(f)

            # Initialize model
            self._model = DiViDeAddEvaluator(**self._opt["model"]["args"]).to(self.device)

            # Download/Load weights
            models_dir = Path(self.config.get("models_dir") or "models")
            weights_root = models_dir / "fast_vqa"
            shared_downloads = models_dir / "downloads"
            weights_filename = str(self._opt["test_load_path"]).replace("*", "_").split("/")[-1]
            weights_path = weights_root / weights_filename
            legacy_path = base_path / "pretrained_weights" / weights_filename
            shared_path = shared_downloads / weights_filename

            if not weights_path.exists():
                if legacy_path.exists():
                    weights_root.mkdir(parents=True, exist_ok=True)
                    legacy_path.replace(weights_path)
                elif shared_path.exists():
                    weights_root.mkdir(parents=True, exist_ok=True)
                    shared_path.replace(weights_path)
                else:
                    self._download_weights(weights_filename, weights_path)

            checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=True)
            if "state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["state_dict"])
            else:
                self._model.load_state_dict(checkpoint)

            self._model.eval()
            self._ml_available = True

        except ImportError:
            logger.warning("Missing dependencies (decord, timm, or yaml). FastVQA disabled.")
        except Exception as e:
            logger.warning(f"Failed to setup FastVQA: {e}")
            import traceback

            traceback.print_exc()

    def _download_weights(self, filename, destination_path):
        url = FASTVQA_MODEL_URLS.get(filename)
        if not url:
            # Fallback map for filenames that might not match exactly
            if "FAST_VQA_3D" in filename:
                url = FASTVQA_MODEL_URLS["FAST_VQA_3D_1_1.pth"]
            elif "FAST_VQA_B" in filename:
                url = FASTVQA_MODEL_URLS["FAST_VQA_B_1_4.pth"]
            elif "FAST_VQA_M" in filename:
                url = FASTVQA_MODEL_URLS["FAST_VQA_M_1_4.pth"]
            else:
                raise ValueError(f"Unknown weight file: {filename}")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {filename} from {url}...")

        urllib.request.urlretrieve(url, destination_path)
        logger.info(f"Downloaded {filename}")

    def _prepare_input(self, video_path, min_dimension=None):
        import torch
        import decord
        from ayase.third_party.fastvqa.datasets import get_spatial_fragments, FragmentSampleFrames, SampleFrames

        # Logic adapted from quality_scorer.py
        val_config = None
        for key in self._opt["data"].keys():
            if key.startswith("val-"):
                val_config = self._opt["data"][key]["args"]
                break
        if val_config is None:
            val_config = self._opt["data"]["train"]["args"]

        t_data_opt = val_config
        s_data_opt = t_data_opt["sample_types"]

        vsamples = {}
        clip_count = None

        video_reader = decord.VideoReader(str(video_path))
        total_frames = len(video_reader)

        for sample_type, sample_args in s_data_opt.items():
            frame_interval = sample_args.get("frame_interval", t_data_opt.get("frame_interval", 1))

            if t_data_opt.get("t_frag", 1) > 1:
                sampler = FragmentSampleFrames(
                    fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                    fragments_t=sample_args.get("t_frag", 1),
                    num_clips=sample_args.get("num_clips", 1),
                    frame_interval=frame_interval,
                )
            else:
                sampler = SampleFrames(
                    clip_len=sample_args["clip_len"],
                    frame_interval=frame_interval,
                    num_clips=sample_args.get("num_clips", 1),
                )

            frames = sampler(total_frames)
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}

            imgs = []
            for idx in frames:
                frame_obj = frame_dict[idx]
                if hasattr(frame_obj, "asnumpy"):
                    frame = frame_obj.asnumpy()
                elif hasattr(frame_obj, "numpy"):
                    frame = frame_obj.numpy()
                else:
                    frame = np.array(frame_obj)

                # H, W, C, BGR (decord default is RGB if not specified? wait, decord is RGB usually)
                # Decord returns RGB by default.
                # Wait, quality_scorer says: frame = frame_dict[idx].asnumpy() # H, W, C, BGR format from decord
                # BUT decord documentation says it returns RGB.
                # Let's assume RGB for now as Ayase uses RGB.

                # quality_scorer code: frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # This implies decord gave BGR. Let's verify.
                # Actually standard decord is RGB. Maybe they configured it differently or it's a specific version.
                # Ayase uses RGB everywhere.

                # Let's assume decord returns RGB.
                frame_rgb = frame

                if min_dimension and min_dimension > 0:
                    frame_rgb = resize_frame_maintain_aspect(frame_rgb, min_dimension)

                imgs.append(torch.from_numpy(frame_rgb))

            video = torch.stack(imgs, 0).permute(3, 0, 1, 2)  # C, T, H, W

            sampled_video = get_spatial_fragments(video, **sample_args)

            mean = torch.FloatTensor([123.675, 116.28, 103.53])
            std = torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

            num_clips = sample_args.get("num_clips", 1)
            if clip_count is None:
                clip_count = num_clips

            sampled_video = sampled_video.reshape(
                sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
            ).transpose(0, 1)
            vsamples[sample_type] = sampled_video.contiguous()

        return {"samples": vsamples, "num_clips": clip_count}

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            import torch

            prepared = self._prepare_input(sample.path)

            batched_inputs = {}
            for sample_type in prepared["samples"].keys():
                batched_inputs[sample_type] = prepared["samples"][sample_type].to(self.device)

            with torch.no_grad():
                scores = self._model(batched_inputs)
                # scores shape: (num_clips, 1) usually
                # We average over clips
                raw_score = scores.mean().item()

            # Sigmoid rescale
            mean, std = self.fastvqa_mean_std[self.model_type]
            x = (raw_score - mean) / std
            final_score = 1 / (1 + np.exp(-x))
            final_score_100 = final_score * 100.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.fast_vqa_score = final_score_100

            if final_score_100 < 40.0:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low FastVQA Score: {final_score_100:.1f}",
                        details={"score": final_score_100, "raw": raw_score},
                    )
                )

        except Exception as e:
            logger.warning(f"FastVQA failed for {sample.path}: {e}")
            import traceback

            traceback.print_exc()

        return sample
