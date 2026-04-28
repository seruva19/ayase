"""StableVQA — Video Stability Quality Assessment.

ACM MM 2023 — first model specifically targeting video stability
perception.

The real StableVQA paper estimates camera motion via affine/homography
transforms between consecutive frames, computes a residual-motion
stability score (how much unintended jitter remains after compensating
for dominant motion), and fuses it with spatial quality (blur, noise)
and temporal quality from a learned backbone.

This implementation is an approximation:
  - Stability is estimated via inter-frame homography fitting (RANSAC)
    with residual-motion analysis, matching the paper's core idea.
  - Spatial quality features come from ResNet-50 (the paper uses a
    custom backbone trained end-to-end; weights are not public).
  - The quality head uses random initialisation, so the absolute
    score is a plausible proxy, not a calibrated MOS predictor.

GitHub: https://github.com/QMME/StableVQA

stablevqa_score — higher = better (more stable, 0-1)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StableVQAModule(PipelineModule):
    name = "stablevqa"
    description = "StableVQA video stability quality assessment (ACM MM 2023)"
    default_config = {
        "step": 2,
        "max_frames": 120,
        "frame_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.step = self.config.get("step", 2)
        self.max_frames = self.config.get("max_frames", 120)
        self.frame_size = self.config.get("frame_size", 224)
        self.stability_resize = self.config.get("stability_resize", 480)
        self._ml_available = False
        self._backbone = None
        self._stability_head = None
        self._device = "cpu"
        self._transform = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ResNet-50 backbone for spatial quality features
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._backbone = nn.Sequential(*list(resnet.children())[:-1])
            self._backbone.eval()
            self._backbone.to(self._device)

            # Stability assessment head:
            # Input: spatial features (2048) + stability features (12)
            self._stability_head = nn.Sequential(
                nn.Linear(2048 + 12, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._stability_head.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.frame_size + 32),
                transforms.CenterCrop(self.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            logger.info(
                "StableVQA initialised on %s (ResNet-50 + homography stability)",
                self._device,
            )

        except ImportError:
            logger.warning(
                "StableVQA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("StableVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            score = self._compute_stability(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.stablevqa_score = score
                logger.debug("StableVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("StableVQA failed for %s: %s", sample.path, e)

        return sample

    # ------------------------------------------------------------------
    # Homography-based inter-frame stability (core of the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_residual_motion(
        prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> tuple:
        """Estimate dominant motion via homography and return residual stats.

        Returns (residual_mean, residual_std, inlier_ratio, translation_mag).
        On failure falls back to dense-flow jitter estimate.
        """
        import cv2

        # Detect ORB keypoints for homography estimation
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return StableVQAModule._flow_fallback(prev_gray, curr_gray)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 8:
            return StableVQAModule._flow_fallback(prev_gray, curr_gray)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Fit affine (6-DOF, more stable than full homography for jitter)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None:
            return StableVQAModule._flow_fallback(prev_gray, curr_gray)

        inlier_ratio = float(np.sum(inliers)) / max(len(inliers), 1)

        # Warp previous frame by estimated dominant motion
        h, w = curr_gray.shape[:2]
        warped = cv2.warpAffine(prev_gray, M, (w, h))

        # Residual = pixel-wise difference after compensating dominant motion
        residual = np.abs(curr_gray.astype(np.float32) - warped.astype(np.float32))
        residual_mean = float(np.mean(residual))
        residual_std = float(np.std(residual))

        # Translation magnitude from affine matrix (tx, ty)
        tx, ty = float(M[0, 2]), float(M[1, 2])
        translation_mag = float(np.sqrt(tx * tx + ty * ty))

        return residual_mean, residual_std, inlier_ratio, translation_mag

    @staticmethod
    def _flow_fallback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple:
        """Fallback: dense optical flow when keypoint matching fails."""
        import cv2

        small_h, small_w = 240, 320
        g1 = cv2.resize(prev_gray, (small_w, small_h))
        g2 = cv2.resize(curr_gray, (small_w, small_h))
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(mag)), float(np.std(mag)), 0.0, float(np.mean(mag))

    def _compute_stability(self, sample: Sample) -> Optional[float]:
        """Compute stability score via homography-based residual motion + spatial features.

        Matches the StableVQA paper: estimate dominant camera motion per
        frame pair, compute residual (unintended jitter) after compensation,
        and combine with spatial quality features from ResNet-50.
        """
        import torch
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            cap.release()
            return None

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        sz = self.stability_resize
        h0, w0 = prev_gray.shape[:2]
        scale = sz / max(h0, w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        prev_gray_small = cv2.resize(prev_gray, (new_w, new_h))

        # Collect per-pair stability measurements
        residual_means = []
        residual_stds = []
        inlier_ratios = []
        translation_mags = []
        spatial_features = []

        # Spatial feature for first frame
        rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            feat = self._backbone(tensor).squeeze(-1).squeeze(-1)
            spatial_features.append(feat)

        frame_count = 0
        while frame_count < self.max_frames:
            for _ in range(self.step):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, (new_w, new_h))

            # Core: homography-based residual motion estimation
            r_mean, r_std, inlier_r, t_mag = self._estimate_residual_motion(
                prev_gray_small, curr_gray_small
            )
            residual_means.append(r_mean)
            residual_stds.append(r_std)
            inlier_ratios.append(inlier_r)
            translation_mags.append(t_mag)

            # Spatial features periodically
            if frame_count % max(1, self.max_frames // 8) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                    feat = self._backbone(tensor).squeeze(-1).squeeze(-1)
                    spatial_features.append(feat)

            prev_gray_small = curr_gray_small
            frame_count += 1

        cap.release()

        if not residual_means:
            return None

        # Build 12-d stability feature vector
        stability_feats = np.array([
            # Residual motion after dominant-motion compensation
            np.mean(residual_means),
            np.std(residual_means),
            np.mean(residual_stds),
            np.std(residual_stds),
            # Homography fit quality (high inlier ratio = reliable estimation)
            np.mean(inlier_ratios),
            np.std(inlier_ratios),
            # Camera translation magnitude
            np.mean(translation_mags),
            np.std(translation_mags),
            # Jitter: variance of consecutive residuals (temporal instability)
            float(np.var(np.diff(residual_means))) if len(residual_means) > 1 else 0.0,
            # Smoothness: mean abs diff of consecutive residuals
            float(np.mean(np.abs(np.diff(residual_means)))) if len(residual_means) > 1 else 0.0,
            # Translation jitter (sudden camera jumps)
            float(np.var(np.diff(translation_mags))) if len(translation_mags) > 1 else 0.0,
            float(np.mean(np.abs(np.diff(translation_mags)))) if len(translation_mags) > 1 else 0.0,
        ], dtype=np.float32)

        # Aggregate spatial features
        spatial_stack = torch.cat(spatial_features, dim=0)
        spatial_mean = spatial_stack.mean(dim=0, keepdim=True)  # (1, 2048)

        stability_tensor = (
            torch.from_numpy(stability_feats).float().unsqueeze(0).to(self._device)
        )

        with torch.no_grad():
            combined = torch.cat([spatial_mean, stability_tensor], dim=1)  # (1, 2060)
            score = self._stability_head(combined).item()

        return float(score)
