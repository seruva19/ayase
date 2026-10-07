"""FloLPIPS — flow-weighted LPIPS (Danier et al., PCS 2022).

Full-reference video quality metric implementing the official protocol
(``flolpips.py`` / ``calc_flolpips`` in github.com/danier97/flolpips):

  * for each consecutive frame pair (t, t+1) of both clips, PWC-Net optical
    flow is computed on the reference pair and the distorted pair, and the
    weight map is the *difference* ``flow_ref - flow_dis``;
  * LPIPS-Alex (v0.1) is evaluated per layer: normalised feature differences
    pass through the linear heads and each layer map is pooled with
    ``mw_spatial_average`` — flow magnitude interpolated to that layer's
    resolution, normalised to sum 1, used as weights;
  * the pair score is the sum over layers; the video score is the mean over
    all consecutive pairs, at native resolution.

PWC-Net is provided through the ``ptlflow`` package (a pure-PyTorch port of
the official model with converted weights); LPIPS-Alex v0.1 through the
``lpips`` package, whose internals reproduce the FloLPIPS layer-wise forward.
When either is missing the metric is left unset — no substitute flow or map
is used.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _mw_spatial_average(in_tens, flow, F, keepdim=True):
    """Verbatim mw_spatial_average from the official flolpips.py."""
    _, _, h, w = in_tens.shape
    flow = F.interpolate(flow, (h, w), align_corners=False, mode="bilinear")
    flow_mag = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)
    flow_mag = flow_mag / torch.sum(flow_mag, dim=[1, 2, 3], keepdim=True)
    return torch.sum(in_tens * flow_mag, dim=[2, 3], keepdim=keepdim)


import torch  # noqa: E402  (needed at module level for _mw_spatial_average)


class FloLPIPSModule(PipelineModule):
    name = "flolpips"
    provenance = "published"
    sources = {
        "flolpips": "FloLPIPS (Danier et al., PCS 2022) — https://github.com/danier97/flolpips",
    }
    deviations = {
        "flolpips": "PWC-Net — pure-torch port with converted official weights (ptlflow) instead of cupy-CUDA correlation; frame-pair counts use min(len_ref, len_dis) instead of asserting equal length",
    }
    description = "Flow-weighted LPIPS full-reference video quality (PWC-Net + LPIPS-Alex)"
    default_config = {}
    metric_groups = {
        "flolpips": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = "unavailable"
        self._lpips_model = None
        self._flownet = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import os
            import torch  # noqa: F401
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ.setdefault("TORCH_HOME", str(models_dir))

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            # LPIPS-Alex v0.1 — required; the forward below uses its
            # scaling_layer / net / lins exactly as the official FloLPIPS.
            import lpips

            def load_lpips():
                return lpips.LPIPS(net="alex", spatial=True).to(self._device).eval()

            self._lpips_model = shared_runtime_resource(
                self,
                ("lpips_alex_spatial", str(self._device)),
                load_lpips,
            )

            # PWC-Net via ptlflow — pure-PyTorch port of the official model.
            import ptlflow

            def load_pwc():
                model = ptlflow.get_model("pwcnet", pretrained_ckpt="things")
                return model.to(self._device).eval()

            self._flownet = shared_runtime_resource(
                self,
                ("pwcnet_things", str(self._device)),
                load_pwc,
            )

            self._backend = "pwcnet_lpips"
            logger.info("FloLPIPS initialised (PWC-Net + LPIPS-Alex) on %s", self._device)
        except ImportError as e:
            logger.warning(
                "FloLPIPS unavailable: requires the 'lpips' and 'ptlflow' packages (%s)", e
            )
        except Exception as e:
            logger.warning("FloLPIPS unavailable: backend load failed (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if self._backend != "pwcnet_lpips" or not sample.is_video:
            return sample

        # FloLPIPS is full-reference; without a reference video there is no metric.
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None or not Path(str(reference_path)).exists():
            return sample

        try:
            ref_frames = self._load_all(Path(str(reference_path)))
            dist_frames = self._load_all(sample.path)
            n = min(len(ref_frames), len(dist_frames))
            if n < 2:
                return sample

            scores = []
            with torch.no_grad():
                for i in range(n - 1):
                    ref_t = self._to_t(ref_frames[i])
                    ref_t1 = self._to_t(ref_frames[i + 1])
                    dis_t = self._to_t(dist_frames[i])
                    dis_t1 = self._to_t(dist_frames[i + 1])
                    flow_ref = self._flow(ref_t, ref_t1)
                    flow_dis = self._flow(dis_t, dis_t1)
                    if flow_ref is None or flow_dis is None:
                        continue
                    flow_diff = flow_ref - flow_dis
                    val = self._flolpips(ref_t, dis_t, flow_diff)
                    if val is not None:
                        scores.append(val)

            if scores:
                sample.quality_metrics.flolpips = float(np.mean(scores))
        except Exception as e:
            logger.warning("FloLPIPS failed: %s", e)
        return sample

    # ------------------------------------------------------------- internals
    def _to_t(self, rgb: np.ndarray):
        import torch

        return (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self._device)
            / 255.0
        )

    def _flow(self, t0, t1):
        """PWC-Net flow between two [1,3,H,W] frames in [0,1] → [1,2,H,W]."""
        import torch

        try:
            inputs = {"images": torch.cat([t0, t1], dim=0).unsqueeze(1)}  # [1,2,3,H,W]
            inputs["images"] = inputs["images"].permute(0, 1, 2, 3, 4)
            preds = self._flownet(inputs)
            flow = preds["flows"]
            # flows: [N, n_preds, 2, H, W] — take the last (finest) prediction
            if flow.ndim == 5:
                flow = flow[:, -1]
            if flow.ndim == 4 and flow.shape[1] != 2:
                flow = flow.squeeze(1)
            return flow[:, :2]
        except Exception as e:
            logger.debug("PWC-Net flow failed: %s", e)
            return None

    def _flolpips(self, ref_t, dis_t, flow_diff) -> Optional[float]:
        """Verbatim FloLPIPS.forward — per-layer LPIPS diffs pooled by
        |flow_ref - flow_dis| magnitude at each layer's resolution."""
        import torch
        import torch.nn.functional as F
        from lpips.lpips import normalize_tensor

        m = self._lpips_model
        in0 = m.scaling_layer(ref_t * 2.0 - 1.0)   # v0.1: normalize=True path
        in1 = m.scaling_layer(dis_t * 2.0 - 1.0)
        outs0, outs1 = m.net.forward(in0), m.net.forward(in1)

        res = []
        for kk in range(m.L):
            f0 = normalize_tensor(outs0[kk])
            f1 = normalize_tensor(outs1[kk])
            diffs = (f0 - f1) ** 2
            res.append(_mw_spatial_average(m.lins[kk](diffs), flow_diff, F))

        return float(torch.sum(torch.cat(res, 1), dim=(1, 2, 3)).item())

    def _load_all(self, path: Path) -> List[np.ndarray]:
        """All consecutive frames at native resolution, RGB uint8."""
        import cv2

        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames
