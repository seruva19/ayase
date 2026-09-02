"""ChronoMagic-Bench module — NeurIPS 2024 (arXiv:2406.18522).

Two metrics for time-lapse / metamorphic video, from the reference
ChronoMagic-Bench (github.com/PKU-YuanGroup/ChronoMagic-Bench):

  - **MTScore** (Metamorphic Temporal score) — ``chronomagic_mt_score``.
    Video-text retrieval with an **InternVideo2-stage2** encoder over ten
    fixed prompts (five "ordinary video", five "time-lapse / metamorphic").
    MTScore is the summed retrieval probability mass assigned to the
    metamorphic prompts among the top-k retrieved. Higher = more metamorphic
    (more time-lapse-like temporal change).

  - **CHScore** (Coherence score, upstream ``TSI_score``) — ``chronomagic_ch_score``.
    **CoTracker2** grid point-tracking; a set of temporal-stability
    statistics on the tracked points (missed-point ratio and its variation,
    frame-cut ratio, continuous change) are min-max normalised, weighted, and
    summed into ``TSI_sum``; CHScore = ``1 / TSI_sum``. Higher = more coherent
    motion (fewer tracking discontinuities / hallucinated frames).

No-heuristic policy: each sub-metric is a real backbone output or ``None``.
CoTracker2 (CHScore) loads from ``torch.hub``; InternVideo2 (MTScore) needs the
vendored ChronoMagic-Bench ``MTScore`` package plus its 1B checkpoint (neither
pip-installable nor in the weight mirror), so MTScore is populated only when a
local InternVideo2 checkout + checkpoint are configured, otherwise left ``None``.
If only one backbone is present, only that sub-metric is emitted; the other stays
``None`` (never fabricated).

Video-only: returns unchanged for images.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# The ten fixed MTScore prompts (verbatim from ChronoMagic-Bench
# MTScore/step0-get_MTScore.py). Indices 0-4 are "ordinary video" (general),
# 5-9 are "time-lapse / metamorphic".
_MT_TEXT_CANDIDATES: List[str] = [
    "A conventional video, not a time-condensed video.",
    "A usual video, not an accelerated video sequence.",
    "A normal video, not a time-lapse video.",
    "A standard video, not a time-lapse.",
    "An ordinary video, different from a fast-motion video.",
    "A time-lapse video, distinct from a regular recording.",
    "A time-lapse footage, not your typical video.",
    "A fast-motion video, unlike a standard video.",
    "A time-condensed video, not a conventional video.",
    "An accelerated video sequence, not a usual video.",
]
_MT_GENERAL_INDICES = {0, 1, 2, 3, 4}

# CHScore normalisation constants (verbatim from
# CHScore/step0-get_CHScore.py :: get_score). These calibrate the raw
# per-video statistics onto a common scale before the weighted sum.
_CH_GLOBAL_STATS = {
    "AMPR_score": {"max": 0.9368749856948853, "min": 0.0},
    "MPVR_score": {"max": 0.8997353911399841, "min": 0.0},
    "FCR_score": {"max": 0.625, "min": 0.0},
    "CMPV_score": {"max": 20, "min": 0},
    "MCMPV_score": {"max": 1.0, "min": 0.0},
}
_CH_WEIGHTS = {
    "CMPV_score": 0.15,
    "FCR_score": 0.15,
    "AMPR_score": 0.35,
    "MPVR_score": 0.25,
    "MCMPV_score": 0.10,
}


class ChronoMagicModule(PipelineModule):
    name = "chronomagic"
    description = "ChronoMagic-Bench MTScore (InternVideo2) + CHScore (CoTracker2)"
    default_config = {
        # CHScore (CoTracker2) — upstream defaults.
        "ch_grid_size": 30,
        "ch_threshold": 0.1,
        "ch_size": None,          # resize shortest edge to this (None = no resize, upstream default)
        "ch_max_frames": None,    # None = all frames (upstream); int caps for memory
        # MTScore (InternVideo2) — real backbone only when configured.
        "internvideo2_repo": None,    # path to a ChronoMagic-Bench/MTScore checkout (has configs/ + models/)
        "internvideo2_config": "configs/internvideo2_stage2_config.py",
        "internvideo2_ckpt": None,    # path to InternVideo2-stage2_1b-224p-f4.pt
        "mt_topk": 5,
    }
    metric_groups = {
        "chronomagic_ch_score": "temporal",
        "chronomagic_mt_score": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._device = "cpu"
        # CHScore / CoTracker2
        self._ch_available = False
        self._cotracker = None
        # MTScore / InternVideo2
        self._mt_available = False
        self._intern_model = None
        self._mt_config = None
        self._retrieve_text = None

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        if self.test_mode:
            return

        self._setup_chscore()
        self._setup_mtscore()

        if self._ch_available or self._mt_available:
            self._backend = "real"
        else:
            self._backend = "unavailable"
            logger.warning(
                "ChronoMagic unavailable: CoTracker2 (CHScore) failed to load and "
                "InternVideo2 (MTScore) is not configured; no scores will be emitted."
            )

    def _setup_chscore(self) -> None:
        """CHScore backbone: CoTracker2 via torch.hub (same path as physics)."""
        try:
            import torch

            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            from ._cotracker_utils import load_cotracker

            self._cotracker = load_cotracker(
                device, str(self.config.get("models_dir", "models"))
            )
            self._device = device
            self._ch_available = True
            logger.info("ChronoMagic CHScore loaded CoTracker2 on %s", device)
        except Exception as e:
            logger.info("CoTracker2 unavailable for ChronoMagic CHScore: %s", e)

    def _setup_mtscore(self) -> None:
        """MTScore backbone: InternVideo2-stage2 from a vendored ChronoMagic-Bench
        MTScore checkout + 1B checkpoint. Real model or None — no substitute."""
        repo = self.config.get("internvideo2_repo")
        ckpt = self.config.get("internvideo2_ckpt")
        if not repo or not ckpt:
            logger.info(
                "ChronoMagic MTScore not configured (need 'internvideo2_repo' + "
                "'internvideo2_ckpt'); mt_score will stay None."
            )
            return

        import os
        import sys

        if not os.path.isdir(repo) or not os.path.isfile(ckpt):
            logger.warning(
                "ChronoMagic MTScore: internvideo2_repo/ckpt path does not exist "
                "(repo=%s, ckpt=%s); mt_score will stay None.",
                repo,
                ckpt,
            )
            return

        try:
            import torch  # noqa: F401

            if repo not in sys.path:
                sys.path.insert(0, repo)
            # Upstream MTScore helpers (models/ + configs/ live under `repo`).
            from configs.config import Config, eval_dict_leaf  # type: ignore
            from configs.utils import retrieve_text, setup_internvideo2  # type: ignore

            cfg_path = self.config.get(
                "internvideo2_config", "configs/internvideo2_stage2_config.py"
            )
            if not os.path.isabs(cfg_path):
                cfg_path = os.path.join(repo, cfg_path)
            config = eval_dict_leaf(Config.from_file(cfg_path))
            config["model"]["vision_encoder"]["pretrained"] = ckpt

            self._intern_model, _ = setup_internvideo2(config)
            self._mt_config = config
            self._retrieve_text = retrieve_text
            self._mt_available = True
            logger.info("ChronoMagic MTScore loaded InternVideo2-stage2 from %s", repo)
        except Exception as e:
            logger.warning(
                "ChronoMagic MTScore: failed to load InternVideo2 from %s: %s; "
                "mt_score will stay None.",
                repo,
                e,
            )

    # ------------------------------------------------------------------ #
    # Process                                                              #
    # ------------------------------------------------------------------ #

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if self._backend != "real":
            return sample
        if not sample.is_video:
            return sample

        if self._ch_available:
            try:
                ch = self._compute_chscore(sample)
                if ch is not None:
                    sample.quality_metrics.chronomagic_ch_score = float(ch)
            except Exception as e:
                logger.warning("ChronoMagic CHScore failed: %s", e)

        if self._mt_available:
            try:
                mt = self._compute_mtscore(sample)
                if mt is not None:
                    sample.quality_metrics.chronomagic_mt_score = float(mt)
            except Exception as e:
                logger.warning("ChronoMagic MTScore failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # CHScore (CoTracker2)                                                 #
    # ------------------------------------------------------------------ #

    def _read_all_frames(self, path, size: Optional[int]) -> Optional[np.ndarray]:
        """Read RGB frames as [T, H, W, 3] uint8.

        Mirrors CHScore/read_video_from_path: optional shortest-edge resize to
        ``size`` (None = keep native). Uses imageio when present, else OpenCV.
        """
        max_frames = self.config.get("ch_max_frames")

        def _resize(arr: np.ndarray) -> np.ndarray:
            if size is None:
                return arr
            h, w = arr.shape[:2]
            if min(h, w) <= size:
                return arr
            scale = size / min(h, w)
            import cv2

            return cv2.resize(arr, (int(w * scale), int(h * scale)))

        frames: List[np.ndarray] = []
        try:
            import imageio.v2 as imageio  # type: ignore

            reader = imageio.get_reader(str(path))
            for im in reader:
                arr = np.asarray(im)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                frames.append(_resize(arr[..., :3]))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            reader.close()
        except Exception:
            import cv2

            cap = cv2.VideoCapture(str(path))
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            cap.release()

        if len(frames) < 2:
            return None
        return np.stack(frames)

    def _compute_chscore(self, sample: Sample) -> Optional[float]:
        import torch

        grid_size = int(self.config.get("ch_grid_size", 30))
        threshold = float(self.config.get("ch_threshold", 0.1))
        size = self.config.get("ch_size")

        frames = self._read_all_frames(sample.path, size)
        if frames is None or len(frames) < 2:
            return None

        video = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self._device)
        )

        with torch.no_grad():
            trackers, pred_visibility = self._cotracker(video, grid_size=grid_size)

        return self._score_tracks(trackers, pred_visibility, threshold)

    @staticmethod
    def _score_tracks(trackers, pred_visibility, threshold: float) -> Optional[float]:
        """Verbatim port of CHScore/step0-get_CHScore.py :: get_score.

        trackers: [1, T, N, 2]; pred_visibility: [1, T, N].
        Returns TSI_score = 1 / TSI_sum (the per-video CHScore).
        """
        import torch

        _, frames, point_num = pred_visibility.shape
        if frames < 2 or point_num < 1:
            return None

        T = trackers.shape[1]
        N = trackers.shape[2]

        # Movement vector: first frame -> middle frame (upstream uses T // 2).
        first_track = trackers[:, 0, :, :].reshape(N, 2)
        last_track = trackers[:, int(T // 2), :, :].reshape(N, 2)
        pos_vector = last_track - first_track

        norms = torch.norm(pos_vector, dim=1, keepdim=True) + 1e-8
        movement_directions = pos_vector / norms
        initial_positions = trackers[0, 0, :, :]

        miss_points = []
        for i in range(frames):
            miss_points_mask = pred_visibility[0, i, :] == 0
            positions = trackers[0, i, :, :]
            delta_positions = positions - initial_positions
            scalar_projections = torch.sum(delta_positions * movement_directions, dim=1)
            far_direction_mask = scalar_projections > 0
            adjusted = miss_points_mask & (~far_direction_mask)
            miss_points.append(adjusted.sum().float() / point_num)

        miss_points = torch.tensor(miss_points)
        miss_points_ap = torch.abs(miss_points[1:] - miss_points[:-1])

        frames_to_be_cut = (miss_points_ap > threshold).nonzero(as_tuple=True)[0] + 1
        frames_to_be_cut = frames_to_be_cut.cpu().tolist()

        raw_scores = {
            "AMPR_score": miss_points.mean().item(),
            "MPVR_score": miss_points_ap.std().item(),
            "FCR_score": len(frames_to_be_cut) / frames,
            "CMPV_score": (miss_points_ap > threshold).sum().item(),
            "MCMPV_score": miss_points_ap.max().item(),
        }

        normalized_scores = {}
        for key, value in raw_scores.items():
            min_val = _CH_GLOBAL_STATS[key]["min"]
            max_val = _CH_GLOBAL_STATS[key]["max"]
            normalized_scores[key] = (
                (value - min_val) / (max_val - min_val) if max_val != min_val else 0
            )

        tsi_sum = sum(normalized_scores[k] * _CH_WEIGHTS[k] for k in normalized_scores)
        return float(1.0 / tsi_sum) if tsi_sum != 0 else 0.0

    # ------------------------------------------------------------------ #
    # MTScore (InternVideo2)                                               #
    # ------------------------------------------------------------------ #

    def _compute_mtscore(self, sample: Sample) -> Optional[float]:
        """Verbatim port of MTScore/step0-get_MTScore.py :: calculate_video_score.

        Retrieves the top-k prompts for the video with InternVideo2, then sums
        the probability mass over the metamorphic (time-lapse) prompts.
        """
        import cv2

        video = cv2.VideoCapture(str(sample.path))
        frames = []
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
        video.release()
        if len(frames) < self._mt_config.get("num_frames", 8):
            return None

        topk = int(self.config.get("mt_topk", 5))
        try:
            device = next(self._intern_model.parameters()).device
        except Exception:
            device = self._device
        texts, probs = self._retrieve_text(
            frames,
            _MT_TEXT_CANDIDATES,
            model=self._intern_model,
            topk=topk,
            config=self._mt_config,
            device=device,
        )

        text_to_index = {t: i for i, t in enumerate(_MT_TEXT_CANDIDATES)}
        metamorphic_prob = 0.0
        for t, p in zip(texts, probs):
            if text_to_index[t] not in _MT_GENERAL_INDICES:
                metamorphic_prob += float(p)
        return metamorphic_prob

    def _compute_metrics(self, sample: Sample) -> Tuple[Optional[float], Optional[float]]:
        """Convenience for direct testing: (mt_score, ch_score)."""
        mt = self._compute_mtscore(sample) if self._mt_available else None
        ch = self._compute_chscore(sample) if self._ch_available else None
        return mt, ch
