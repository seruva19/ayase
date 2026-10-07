"""EvalCrafter warping error — motion-compensated residual error for frame pairs.

RAFT (raft-things weights, iters=20) computes bidirectional flow on
half-resolution frames and the score is the mean occlusion-masked RGB
residual after warping. Every frame pair is used (no striding). If RAFT is
unavailable the module emits no score. Higher ``warping_error`` means more
unexplained frame change, not better quality. Motion boundaries, cuts,
lighting changes, and flow failure can resemble flicker.
"""

import logging
import cv2
import numpy as np
from typing import List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TemporalFlickeringModule(PipelineModule):
    name = "temporal_flickering"
    provenance = "adapted"
    sources = {
        "warping_error": "Warping error, EvalCrafter (Liu et al. CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/RAFT/optical_flow_scores.py",
    }
    deviations = {
        "warping_error": "RAFT input normalization is correct here; upstream optical_flow_scores.py divides frames by 255 before the vendored RAFT (which itself expects [0,255]) — their released values are computed on degenerate ~constant input",
    }
    description = "Warping Error using RAFT optical flow with occlusion masking (EvalCrafter)"

    default_config = {
        "warning_threshold": 0.02,
        "max_frames": 0,
        "pair_chunk": 8,
    }
    metric_groups = {
        "warping_error": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 0.02)
        # EvalCrafter evaluates every consecutive pair; max_frames <= 0 keeps
        # that, a positive value is an explicit non-default stride cap.
        self.max_frames = self.config.get("max_frames", 0)
        self.pair_chunk = self.config.get("pair_chunk", 8)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import os
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            def load_raft():
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

                # C_T_V1 is torchvision's port of the official raft-things.pth
                # (Chairs -> FlyingThings3D) — the EvalCrafter checkpoint.
                weights = Raft_Large_Weights.C_T_V1
                model = raft_large(weights=weights, progress=False).to(self._device)
                model.eval()
                return model

            logger.info("Setting up RAFT (things weights) for warping error on %s...", self._device)
            self._model = shared_runtime_resource(
                self,
                ("raft", "raft_large_things", str(self._device)),
                load_raft,
            )
            self._ml_available = True
            self._backend = "raft_large"

        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"RAFT unavailable — warping_error disabled: {e}")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        self._analyze_raft(sample)
        return sample

    def _analyze_raft(self, sample: Sample) -> None:
        """RAFT-based warping error (matches EvalCrafter implementation).

        Frames are decoded in a streaming fashion (bounded memory) and
        consecutive pairs are batched through RAFT in chunks; occlusion-masked
        MSE is accumulated on-device with a single host sync at the end.
        """
        import torch

        try:
            err_sum = 0.0
            n_pairs = 0
            with torch.no_grad():
                for img1_batch, img2_batch in self._iter_pair_batches(sample):
                    batch_err = self._process_pair_batch(img1_batch, img2_batch)
                    if batch_err is None:
                        continue
                    err_sum += float(batch_err.sum().item())
                    n_pairs += batch_err.numel()

            if n_pairs == 0:
                return

            warping_error = err_sum / n_pairs

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.warping_error = float(warping_error)

            if warping_error > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High flickering detected (Warping Error): {warping_error:.4f}",
                        details={"warping_error": float(warping_error)},
                    )
                )

        except Exception as e:
            logger.warning(f"RAFT warping error failed: {e}")

    def _iter_pair_batches(self, sample: Sample):
        """Yield (img1_batch, img2_batch) tensors of consecutive frame pairs.

        Frames are read one at a time (optionally strided to respect
        ``max_frames``), so at most ``pair_chunk`` frame pairs are resident in
        memory at once instead of the whole clip.
        """
        import torch
        import torch.nn.functional as F

        chunk = max(1, int(self.pair_chunk))
        cap = None
        try:
            cap = cv2.VideoCapture(str(sample.path))
            if not cap.isOpened():
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = 1
            if self.max_frames > 0 and total > self.max_frames:
                stride = max(1, total // self.max_frames)

            prev_t = None
            buf1: List[torch.Tensor] = []
            buf2: List[torch.Tensor] = []
            idx = 0
            kept = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                idx += 1
                kept += 1
                if self.max_frames > 0 and kept > self.max_frames:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # [1,3,H,W] float in [0,1], downsampled 2x (matches EvalCrafter)
                t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
                t = t.to(self._device).float() / 255.0
                t = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)

                if prev_t is not None:
                    buf1.append(prev_t)
                    buf2.append(t)
                    if len(buf1) >= chunk:
                        yield torch.cat(buf1, dim=0), torch.cat(buf2, dim=0)
                        buf1, buf2 = [], []
                prev_t = t

            if buf1:
                yield torch.cat(buf1, dim=0), torch.cat(buf2, dim=0)
        finally:
            if cap is not None:
                cap.release()

    def _process_pair_batch(self, img1: "object", img2: "object") -> Optional["object"]:
        """Compute occlusion-masked warping error for a batch of frame pairs.

        img1/img2: [N,3,h,w] float in [0,1] (already downsampled). Returns a
        [N] tensor of per-pair errors, using OOM-halving to bound GPU memory.
        """
        import torch
        import torch.nn.functional as F

        n = img1.shape[0]
        if n == 0:
            return None

        try:
            _, _, h, w = img1.shape
            # RAFT InputPadder pads replicated borders to multiples of 8;
            # torchvision RAFT additionally requires >=128px inputs, so small
            # frames get padded up to that floor (flows are cropped back).
            pad_h = max((8 - h % 8) % 8, 128 - h)
            pad_w = max((8 - w % 8) % 8, 128 - w)
            if pad_h > 0 or pad_w > 0:
                img1p = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
                img2p = F.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")
            else:
                img1p, img2p = img1, img2

            # torchvision RAFT expects [-1,1] inputs; EvalCrafter feeds [0,1]
            # to the original RAFT, which normalizes identically inside.
            img1_t = img1p * 2.0 - 1.0
            img2_t = img2p * 2.0 - 1.0

            fw_flow = self._model(img1_t, img2_t, num_flow_updates=20)[-1]
            bw_flow = self._model(img2_t, img1_t, num_flow_updates=20)[-1]

            if pad_h > 0 or pad_w > 0:
                fw_flow = fw_flow[:, :, :h, :w]
                bw_flow = bw_flow[:, :, :h, :w]

            warped_img2 = self._warp(img2, fw_flow)
            occ = self._detect_occlusion(fw_flow, bw_flow)
            noc = 1.0 - occ  # [N,1,h,w]

            diff_sq = ((warped_img2 - img1) * noc) ** 2  # [N,C,h,w]
            _, c, hh, ww = diff_sq.shape
            per_pair_sq = diff_sq.sum(dim=[1, 2, 3])         # [N]
            per_pair_npix = noc.sum(dim=[1, 2, 3])           # [N] (counts h*w positions)
            denom = torch.where(
                per_pair_npix > 0,
                per_pair_npix,
                torch.full_like(per_pair_npix, float(c * hh * ww)),
            )
            return per_pair_sq / denom

        except RuntimeError as exc:
            if n > 1 and "out of memory" in str(exc).lower():
                if self._device != "cpu":
                    torch.cuda.empty_cache()
                mid = n // 2
                left = self._process_pair_batch(img1[:mid], img2[:mid])
                right = self._process_pair_batch(img1[mid:], img2[mid:])
                parts = [p for p in (left, right) if p is not None]
                if not parts:
                    return None
                return torch.cat(parts, dim=0)
            raise

    def _warp(self, img, flow):
        """Warp a batch of images by optical flow via grid_sample.

        img: [N,C,H,W], flow: [N,2,H,W].
        """
        import torch
        import torch.nn.functional as F

        n, _, h, w = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=torch.float32),
            torch.arange(w, device=flow.device, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = grid_x.unsqueeze(0) + flow[:, 0]
        grid_y = grid_y.unsqueeze(0) + flow[:, 1]
        grid_x = 2.0 * grid_x / (w - 1) - 1.0
        grid_y = 2.0 * grid_y / (h - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [N,H,W,2]

        return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _detect_occlusion(self, fw_flow, bw_flow):
        """EvalCrafter occlusion mask (batched): forward-backward flow
        inconsistency OR motion boundary, per warp_utils.detect_occlusion.

        mask1: |fw + warp(bw)|² > 0.01·(|warp(bw)|² + |fw|²) + 0.5
        mask2: |∇fw|² > 0.01·|fw|² + 0.002 (forward finite differences)
        All magnitudes are squared norms, matching the upstream code.
        """
        import torch
        import torch.nn.functional as F

        n, _, h, w = fw_flow.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=fw_flow.device, dtype=torch.float32),
            torch.arange(w, device=fw_flow.device, dtype=torch.float32),
            indexing="ij",
        )
        map_x = grid_x.unsqueeze(0) + fw_flow[:, 0]
        map_y = grid_y.unsqueeze(0) + fw_flow[:, 1]
        map_x = 2.0 * map_x / (w - 1) - 1.0
        map_y = 2.0 * map_y / (h - 1) - 1.0
        grid = torch.stack([map_x, map_y], dim=-1)  # [N,H,W,2]

        warped_bw = F.grid_sample(bw_flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        # mask1: forward-backward consistency, relative + absolute threshold
        fb = fw_flow + warped_bw
        fb_mag_sq = fb[:, 0] ** 2 + fb[:, 1] ** 2
        ref_mag_sq = (warped_bw[:, 0] ** 2 + warped_bw[:, 1] ** 2) + (
            fw_flow[:, 0] ** 2 + fw_flow[:, 1] ** 2
        )
        mask1 = fb_mag_sq > 0.01 * ref_mag_sq + 0.5

        # mask2: motion boundary — squared gradient magnitude of fw_flow
        u, v = fw_flow[:, 0], fw_flow[:, 1]
        du_dx = torch.zeros_like(u)
        du_dx[:, :, :-1] = u[:, :, :-1] - u[:, :, 1:]
        du_dy = torch.zeros_like(u)
        du_dy[:, :-1, :] = u[:, :-1, :] - u[:, 1:, :]
        dv_dx = torch.zeros_like(v)
        dv_dx[:, :, :-1] = v[:, :, :-1] - v[:, :, 1:]
        dv_dy = torch.zeros_like(v)
        dv_dy[:, :-1, :] = v[:, :-1, :] - v[:, 1:, :]
        grad_sq = du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
        mask2 = grad_sq > 0.01 * (u**2 + v**2) + 0.002

        occ = (mask1 | mask2).float().unsqueeze(1)  # [N,1,H,W]
        return occ
