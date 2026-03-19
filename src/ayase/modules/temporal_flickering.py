import logging
import cv2
import numpy as np
from typing import List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TemporalFlickeringModule(PipelineModule):
    name = "temporal_flickering"
    description = "Warping Error using RAFT optical flow with occlusion masking"

    default_config = {
        "warning_threshold": 0.02,
        "max_frames": 300,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 0.02)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transforms = None

    def setup(self) -> None:
        try:
            import os
            import torch
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up RAFT-Small for warping error on {self._device}...")

            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights, progress=False).to(self._device)
            self._model.eval()
            self._transforms = weights.transforms()
            self._ml_available = True

        except ImportError:
            logger.warning("torchvision >= 0.13 required for RAFT.")
        except Exception as e:
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

        For each consecutive pair:
        1. Compute forward and backward flow with RAFT (downsampled 2x)
        2. Detect occlusions via forward-backward consistency
        3. Warp frame2 to frame1 using forward flow
        4. MSE in non-occluded regions
        """
        import torch
        import torch.nn.functional as F

        try:
            frames = self._load_all_frames(sample)
            if len(frames) < 2:
                return

            total_err = 0.0

            with torch.no_grad():
                for i in range(len(frames) - 1):
                    # Convert to tensor [1, C, H, W] float [0, 255]
                    img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0
                    img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0

                    # Downsample by 2x (matches EvalCrafter)
                    img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
                    img2 = F.interpolate(img2, scale_factor=0.5, mode="bilinear", align_corners=False)

                    # Pad to multiple of 8 (RAFT requirement)
                    _, _, h, w = img1.shape
                    pad_h = (8 - h % 8) % 8
                    pad_w = (8 - w % 8) % 8
                    if pad_h > 0 or pad_w > 0:
                        img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="reflect")
                        img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode="reflect")

                    # Prepare for RAFT (needs [0, 255] range after transforms)
                    img1_raft = img1 * 255.0
                    img2_raft = img2 * 255.0

                    if self._transforms:
                        img1_t, img2_t = self._transforms(img1_raft, img2_raft)
                    else:
                        img1_t, img2_t = img1_raft, img2_raft

                    # Forward flow (frame1 → frame2)
                    fw_flow = self._model(img1_t, img2_t)[-1]
                    # Backward flow (frame2 → frame1)
                    bw_flow = self._model(img2_t, img1_t)[-1]

                    # Crop back to original size (remove padding)
                    if pad_h > 0 or pad_w > 0:
                        crop_h = fw_flow.shape[2] - pad_h if pad_h > 0 else fw_flow.shape[2]
                        crop_w = fw_flow.shape[3] - pad_w if pad_w > 0 else fw_flow.shape[3]
                        fw_flow = fw_flow[:, :, :crop_h, :crop_w]
                        bw_flow = bw_flow[:, :, :crop_h, :crop_w]
                        img1 = img1[:, :, :crop_h, :crop_w]
                        img2 = img2[:, :, :crop_h, :crop_w]

                    # Warp img2 to img1 using forward flow
                    warped_img2 = self._warp(img2, fw_flow)

                    # Detect occlusions via forward-backward consistency
                    occ_mask = self._detect_occlusion(fw_flow, bw_flow)
                    noc_mask = 1.0 - occ_mask

                    # MSE in non-occluded regions
                    diff = (warped_img2 - img1) * noc_mask
                    diff_sq = diff ** 2

                    n_pixels = torch.sum(noc_mask)
                    if n_pixels == 0:
                        n_pixels = torch.tensor(diff_sq.numel(), dtype=torch.float32)

                    total_err += (torch.sum(diff_sq) / n_pixels).item()

            warping_error = total_err / (len(frames) - 1)

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

    def _warp(self, img, flow):
        """Warp image using optical flow via grid_sample."""
        import torch
        import torch.nn.functional as F

        _, _, h, w = img.shape
        # Create normalized grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=torch.float32),
            torch.arange(w, device=flow.device, dtype=torch.float32),
            indexing="ij",
        )
        # Add flow displacement
        grid_x = grid_x + flow[0, 0]
        grid_y = grid_y + flow[0, 1]
        # Normalize to [-1, 1]
        grid_x = 2.0 * grid_x / (w - 1) - 1.0
        grid_y = 2.0 * grid_y / (h - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _detect_occlusion(self, fw_flow, bw_flow, threshold=1.0):
        """Detect occlusions via forward-backward flow consistency."""
        import torch

        # Warp backward flow using forward flow
        _, _, h, w = fw_flow.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=fw_flow.device, dtype=torch.float32),
            torch.arange(w, device=fw_flow.device, dtype=torch.float32),
            indexing="ij",
        )
        map_x = grid_x + fw_flow[0, 0]
        map_y = grid_y + fw_flow[0, 1]
        # Normalize
        map_x = 2.0 * map_x / (w - 1) - 1.0
        map_y = 2.0 * map_y / (h - 1) - 1.0
        grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0)

        import torch.nn.functional as F
        warped_bw = F.grid_sample(bw_flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        # Check consistency: ||fw + warped_bw|| should be small for non-occluded
        consistency = fw_flow + warped_bw
        mag = torch.sqrt(consistency[:, 0] ** 2 + consistency[:, 1] ** 2)
        occ = (mag > threshold).float().unsqueeze(1)

        return occ

    def _analyze_farneback_fallback(self, sample: Sample) -> None:
        """Fallback: Farneback-based warping error when RAFT is unavailable."""
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        prev_gray = None
        warping_errors = []
        frame_idx = 0

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
            frame_idx += 1

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

    def _load_all_frames(self, sample: Sample) -> List[np.ndarray]:
        max_frames = self.config.get("max_frames", 300)
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total > max_frames and total > 0:
                # Subsample uniformly to stay within memory budget
                indices = set(np.linspace(0, total - 1, max_frames, dtype=int))
                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    idx += 1
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames: {e}")
        return frames
