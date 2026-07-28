"""SAMA — Scaling and Masking for Video Quality Assessment (AAAI 2024).

Real implementation of SAMA (https://github.com/Sissuire/SAMA). SAMA builds on
FAST-VQA: it reuses FAST-VQA's Video-Swin-Tiny (group-attention "grpb") backbone
and ``VQAHead`` (imported from :mod:`ayase.third_party.fastvqa`); the SAMA
contribution is the *scaling + masking* spatial-temporal sample construction that
replaces FAST-VQA's plain fragment sampling.

The published LSVQ baseline checkpoint (``SAMA-baseline_val-ltest_s_dev_v0.0.pth``)
is a ``DiViDeAddEvaluator`` with a single ``fragments`` branch and a shared
``vqa_head`` (backbone_size=swin_tiny_grpb, divide_head=false, in_channels=768).
The checkpoint is a dict with ``state_dict`` (197 tensors: ``fragments_backbone.*``
+ ``vqa_head.*``) and ``validation_results``.

Only the real trained SAMA model produces ``sama_score`` (raw model output, an
unbounded quality score where higher = better). No-heuristic policy: real value
or ``None``.
"""

import logging

import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Original weights (GoogleDrive/BaiDu in upstream repo), mirrored on HF.
SAMA_REPO_ID = "AkaneTendo25/ayase-runtime-assets"
SAMA_WEIGHTS_FILE = "sama/SAMA-baseline_val-ltest_s_dev_v0.0.pth"

# Normalisation constants (Kinetics / FAST-VQA / SAMA shared).
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)


class _FragmentSampleFrames:
    """SAMA/FAST-VQA temporal fragment sampler (test mode, deterministic).

    Adopted verbatim from SAMA ``fastvqa/datasets/fusion_datasets.py``. Note the
    upstream positional call ``FragmentSampleFrames(clip_len, num_clips,
    frame_interval)`` maps to ``(fsize_t, fragments_t, frame_interval)`` — that
    quirk is preserved so the sampled clip matches what the checkpoint was
    evaluated with.
    """

    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_frame_indices(self, num_frames):
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t
        rnd_t = np.zeros(len(tgrids), dtype=np.int32) + max(
            0, tlength - self.fsize_t * self.frame_interval
        ) // 2
        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )
        return np.concatenate(ranges_t)

    def __call__(self, total_frames, start_index=0):
        frame_inds = []
        for _ in range(self.num_clips):
            frame_inds += [self._get_frame_indices(total_frames)]
        frame_inds = np.concatenate(frame_inds) + start_index
        frame_inds[frame_inds > total_frames - 1] = total_frames - 1
        return frame_inds.astype(np.int32)


def _get_spatial_sama_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
):
    """SAMA "scaling" spatial sample construction (test mode, deterministic).

    Adopted from SAMA ``get_spatial_sama_fragments`` (fusion_datasets.py). Builds
    a fragment map where consecutive frame-pairs are drawn from progressively
    down-scaled versions of the frame (the "scaling" pyramid), stitched into a
    ``size_h x size_w`` grid of ``fsize`` fragments. ``is_train=False`` path:
    fixed centre offsets, testing ``multiply=4``.
    """
    import torch

    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    multiply = 4  # testing
    step = aligned // 2

    video = video / 255.0
    if ratio < 1:
        res_h, res_w = round(res_h / ratio), round(res_w / ratio)
        video = torch.nn.functional.interpolate(
            video, size=(res_h, res_w), mode="bilinear", align_corners=False
        )
        factors = [1] * 16 * multiply
    else:
        factors = list(np.linspace(1, 1 / ratio, 16)) * multiply

    assert dur_t % aligned == 0, "clip length must be a multiple of aligned"
    size = (size_h, size_w)

    img_scale, hgrids, wgrids = [], [], []
    rnd_h, rnd_w = [], []
    rnd_rh = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5
    rnd_rw = torch.ones((fragments_h, fragments_w, dur_t // aligned)) * 0.5

    for i, scale in enumerate(factors):
        this_h, this_w = round(res_h * scale), round(res_w * scale)
        img_scale.append(
            255.0
            * torch.nn.functional.interpolate(
                video[:, 2 * i : 2 * (i + 1)],
                size=(this_h, this_w),
                mode="bilinear",
                align_corners=False,
            )
        )
        hgrids.append(
            torch.LongTensor(
                [min(this_h // fragments_h * k, this_h - fsize_h) for k in range(fragments_h)]
            )
        )
        wgrids.append(
            torch.LongTensor(
                [min(this_w // fragments_w * k, this_w - fsize_w) for k in range(fragments_w)]
            )
        )
        hlength, wlength = this_h // fragments_h, this_w // fragments_w
        rnd_h.append((rnd_rh[:, :, i // step] * (hlength - fsize_h)).int())
        rnd_w.append((rnd_rw[:, :, i // step] * (wlength - fsize_w)).int())

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    for k, scale in enumerate(factors):
        for i, hs in enumerate(hgrids[k]):
            for j, ws in enumerate(wgrids[k]):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so = hs + rnd_h[k][i][j]
                h_eo = h_so + fsize_h
                w_so = ws + rnd_w[k][i][j]
                w_eo = w_so + fsize_w
                target_video[:, 2 * k : 2 * (k + 1), h_s:h_e, w_s:w_e] = img_scale[k][
                    :, :, h_so:h_eo, w_so:w_eo
                ]

    return target_video


class SAMAModule(PipelineModule):
    name = "sama"
    description = "SAMA scaling+masking VQA (AAAI 2024, real model only)"
    default_config = {
        # SAMA LSVQ baseline test config (fast-SAMA-test.yml).
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 32,
        "clip_len": 32,
        "num_clips": 4,
        "frame_interval": 2,
        "device": "auto",
    }
    models = [
        {
            "id": SAMA_WEIGHTS_FILE,
            "type": "local",
            "url": "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/sama/SAMA-baseline_val-ltest_s_dev_v0.0.pth",
            "task": "SAMA LSVQ baseline video quality checkpoint",
        },
    ]
    metric_info = {
        "sama_score": "SAMA raw video quality score (unbounded, higher = better)",
    }
    metric_groups = {
        "sama_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device = "cpu"
        self._model = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            self._backend = "unavailable"
            return
        try:
            import torch  # noqa: F401
            import decord  # noqa: F401
            from huggingface_hub import hf_hub_download
            from ayase.third_party.fastvqa.models import DiViDeAddEvaluator
            from ayase.runtime import resolve_torch_device

            self.device = resolve_torch_device(self.config.get("device", "auto"))

            # Single-branch fragments model matching the LSVQ baseline checkpoint.
            model_args = dict(
                backbone=dict(fragments=dict(checkpoint=False, pretrained=None)),
                backbone_size="swin_tiny_grpb",
                backbone_preserve_keys="fragments",
                divide_head=False,
                vqa_head=dict(in_channels=768, hidden_channels=64),
            )
            model = DiViDeAddEvaluator(**model_args).to(self.device)

            weights_path = hf_hub_download(
                repo_id=SAMA_REPO_ID, filename=SAMA_WEIGHTS_FILE
            )
            checkpoint = torch.load(
                weights_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            self._model = model
            self._ml_available = True
            self._backend = "real"
            logger.info(f"SAMA (real) initialised on {self.device}")

        except ImportError as e:
            self._backend = "unavailable"
            logger.warning(
                f"SAMA: dependencies missing ({e}); sama_score left unset. "
                "Requires torch, decord, timm, einops, huggingface_hub."
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"SAMA: setup failed ({e}); sama_score left unset.")

    def _prepare_input(self, video_path):
        import torch
        import decord

        # decord's bridge is a PROCESS-GLOBAL setting; leaving it on "torch"
        # breaks every other module that reads via .asnumpy(). Set it only for
        # our own reads and restore the "native" default (what all peers expect).
        decord.bridge.set_bridge("torch")
        try:
            vr = decord.VideoReader(str(video_path))
            total = len(vr)
            if total < 1:
                raise ValueError("empty video")

            sampler = _FragmentSampleFrames(
                self.config.get("clip_len", 32),
                self.config.get("num_clips", 4),
                self.config.get("frame_interval", 2),
            )
            frame_inds = sampler(total)
            frame_dict = {int(idx): vr[int(idx)] for idx in np.unique(frame_inds)}
            imgs = [frame_dict[int(idx)] for idx in frame_inds]
            video = torch.stack(imgs, 0).permute(3, 0, 1, 2).float()  # C, T, H, W
        finally:
            decord.bridge.set_bridge("native")

        sampled = _get_spatial_sama_fragments(
            video,
            fragments_h=self.config.get("fragments_h", 7),
            fragments_w=self.config.get("fragments_w", 7),
            fsize_h=self.config.get("fsize_h", 32),
            fsize_w=self.config.get("fsize_w", 32),
            aligned=self.config.get("aligned", 32),
        )

        mean = torch.FloatTensor(_MEAN)
        std = torch.FloatTensor(_STD)
        sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        # Reshape [C, T, H, W] into num_clips clips of (T // num_clips) frames.
        num_clips = self.config.get("num_clips", 4)
        c, t, h, w = sampled.shape
        clips = (
            sampled.reshape(1, c, t, h, w)
            .reshape(1, c, num_clips, t // num_clips, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(num_clips, c, t // num_clips, h, w)
        )
        return clips.contiguous().to(self.device)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            import torch

            clips = self._prepare_input(sample.path)
            with torch.no_grad():
                out = self._model({"fragments": clips})
            score = float(out.mean().item())

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.sama_score = score

            logger.debug(f"SAMA for {sample.path.name}: {score:.4f}")

        except Exception as e:
            logger.warning(f"SAMA failed for {sample.path}: {e}")

        return sample
