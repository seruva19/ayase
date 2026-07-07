"""Temporal flickering detection via RAFT optical flow warping error with occlusion masking.

Computes forward-backward flow consistency to detect occlusions, then measures
MSE in non-occluded regions. Falls back to Farneback when RAFT unavailable.
Returns warping_error."""

import logging
import cv2
import numpy as np
from typing import List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TemporalFlickeringModule(PipelineModule):
    name = "temporal_flickering"
    description = "Warping Error using RAFT optical flow with occlusion masking"

    default_config = {
        "warning_threshold": 0.02,
        "max_frames": 300,
        "pair_chunk": 8,
    }
    metric_groups = {
        "warping_error": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 0.02)
        self.max_frames = self.config.get("max_frames", 300)
        self.pair_chunk = self.config.get("pair_chunk", 8)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transforms = None
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
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

                weights = Raft_Small_Weights.DEFAULT
                model = raft_small(weights=weights, progress=False).to(self._device)
                model.eval()
                return model, weights.transforms()

            logger.info("Setting up RAFT-Small for warping error on %s...", self._device)
            self._model, self._transforms = shared_runtime_resource(
                self,
                ("raft", "raft_small", str(self._device)),
                load_raft,
            )
            self._ml_available = True
            self._backend = "raft_small"

        except ImportError:
            self._backend = "farneback"
            logger.warning("torchvision >= 0.13 required for RAFT; using Farneback fallback.")
        except Exception as e:
            self._backend = "farneback"
            logger.warning(f"Failed to setup RAFT: {e}")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if self._ml_available:
            self._analyze_raft(sample)
        else:
            self._analyze_farneback_fallback(sample)

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
            self._analyze_farneback_fallback(sample)

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
            if total > self.max_frames and self.max_frames > 0:
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
                if kept > self.max_frames:
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
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img1p = F.pad(img1, (0, pad_w, 0, pad_h), mode="reflect")
                img2p = F.pad(img2, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                img1p, img2p = img1, img2

            # RAFT weight transforms expect images in [0,1] (their
            # convert_image_dtype does NOT rescale already-float inputs); feed
            # the [0,1] tensors directly rather than pre-scaling to [0,255].
            if self._transforms:
                img1_t, img2_t = self._transforms(img1p, img2p)
            else:
                img1_t, img2_t = img1p, img2p

            fw_flow = self._model(img1_t, img2_t)[-1]
            bw_flow = self._model(img2_t, img1_t)[-1]

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

    def _detect_occlusion(self, fw_flow, bw_flow, threshold=1.0):
        """Detect occlusions via forward-backward flow consistency (batched)."""
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

        consistency = fw_flow + warped_bw
        mag = torch.sqrt(consistency[:, 0] ** 2 + consistency[:, 1] ** 2)  # [N,H,W]
        occ = (mag > threshold).float().unsqueeze(1)  # [N,1,H,W]

        return occ

    def _analyze_farneback_fallback(self, sample: Sample) -> None:
        """Fallback: Farneback-based warping error when RAFT is unavailable."""
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        prev_gray = None
        warping_errors = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                h, w = gray.shape
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (grid_x + flow[..., 0]).astype(np.float32)
                map_y = (grid_y + flow[..., 1]).astype(np.float32)
                warped_prev = cv2.remap(prev_gray, map_x, map_y, cv2.INTER_LINEAR)

                diff = (gray.astype(np.float32) / 255.0 - warped_prev.astype(np.float32) / 255.0) ** 2
                warping_errors.append(np.mean(diff))

            prev_gray = gray

        cap.release()

        if not warping_errors:
            return

        avg_error = float(np.mean(warping_errors))
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.warping_error = avg_error

        if avg_error > self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High flickering detected (Warping Error, Farneback fallback): {avg_error:.4f}",
                    details={"warping_error": avg_error},
                )
            )
