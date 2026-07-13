"""MD-VQA — Multi-Dimensional Quality Assessment for UGC Live Videos (CVPR 2023).

MD-VQA assesses UGC live-streaming video quality from three branches that are
fused into a single quality prediction:

* **Semantic branch** — a torchvision ``efficientnet_v2_s`` (ImageNet-1k)
  backbone. The multi-level block features (stages 2-6) are global-average and
  concatenated into a 656-d per-frame descriptor.
* **Distortion branch** — six hand-crafted per-frame distortion metrics
  (sharpness, noise, blockiness, over-/under-exposure, colorfulness), computed
  exactly as upstream.
* **Motion branch** — a torchvision ``r2plus1d_18`` (Kinetics-400) backbone;
  the 512-d avg-pool feature of each 16-frame clip.

The trained fusion + quality head (``MDVQA_cvpr``, ~4.7 MB) consumes the three
branch features and regresses one score. The two backbones are the stock
torchvision pretrained weights (the upstream ``data/*.pth`` files are literally
``efficientnet_v2_s-dd5fe13b.pth`` and ``r2plus1d_18-91a641e6.pth``); only the
fusion/head checkpoint is model-specific and is mirrored on the Hugging Face
Hub. The checkpoint holds *only* the fusion/head (32 tensors) and loads strictly
into ``MDVQA_cvpr``.

MD-VQA's released head emits a single *fused* multi-dimensional quality score,
not separable per-dimension scores, so ayase exposes one field ``mdvqa_score``
(normalised to 0-1, higher = better). Only the trained model produces it — if
the head weights or any dependency (torch / torchvision / opencv) are missing,
the field is left ``None`` (no heuristic).

Upstream: https://github.com/kunyou99/MD-VQA_cvpr2023
(model / preprocessing mirrored from ``video_quality/model.py``,
``cal_quality_metric.py``, ``read_video.py`` and ``video_quality.py``).

mdvqa_score — fused MD-VQA quality (higher = better, 0-1); real model only
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HF_REPO = "AkaneTendo25/ayase-models"
_HF_HEAD = "mdvqa/LSVQ_rp0.pth"  # default: LSVQ-trained fusion/head

# ImageNet normalisation (semantic + motion branches, per upstream video_transform).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted distortion metrics — ported verbatim from upstream
# video_quality/{sharpness,blockinessII,colorfulness,noise,exposure_mul}.py.
# Six per-frame features in the order the model expects:
#   [sharpness, noise, blockiness, over_exposure, under_exposure, colorfulness]
# ─────────────────────────────────────────────────────────────────────────────
def _sharpness(img_gray):
    import torch
    import torch.nn.functional as F

    device = img_gray.device
    F1 = torch.tensor([[1, -1], [0, 0]]).float().unsqueeze(0).unsqueeze(0).to(device)
    F2 = torch.transpose(F1, 2, 3).to(device)
    g_in = img_gray.unsqueeze(0).unsqueeze(0)
    H1 = F.conv2d(g_in, F1).squeeze(0).squeeze(0)
    H2 = F.conv2d(g_in, F2).squeeze(0).squeeze(0)
    g = torch.sqrt(H1 ** 2 + H2 ** 2)
    row, col = g.shape
    B = round(min(row, col) / 16)
    g_center = g[B:-B, B:-B] if B > 0 else g
    MaxG, MinG, MeanG = torch.max(g_center), torch.min(g_center), torch.mean(g_center)
    if MeanG == 0:
        return torch.tensor(0.0)
    CVG = (MaxG - MinG) / MeanG
    return MaxG ** 0.61 * CVG ** 0.39


def _blockinessII(img_gray):
    import torch

    h, w = img_gray.shape
    d_h = img_gray[:, 1:w] - img_gray[:, 0:w - 1]
    B_h = torch.mean(torch.abs(d_h[:, 7:8 * (int(w / 8) - 1):8]))
    A_h = (8 * torch.mean(torch.abs(d_h)) - B_h) / 7
    sig_h = torch.sign(d_h)
    Z_h = torch.sum(torch.mul(sig_h[:, 0:w - 2], sig_h[:, 1:w - 1]) < 0) / (
        sig_h[:, 0:w - 2].shape[0] * sig_h[:, 0:w - 2].shape[1]
    )
    d_v = img_gray[1:h, :] - img_gray[0:h - 1, :]
    B_v = torch.mean(torch.abs(d_v[7:8 * (int(h / 8) - 1):8, :]))
    A_v = (8 * torch.mean(torch.abs(d_v)) - B_v) / 7
    sig_v = torch.sign(d_v)
    Z_v = torch.sum(torch.mul(sig_v[0:h - 2, :], sig_v[1:h - 1, :]) < 0) / (
        sig_v[0:h - 2, :].shape[0] * sig_v[0:h - 2, :].shape[1]
    )
    B = (B_h + B_v) / 2
    A = (A_h + A_v) / 2
    Z = (Z_h + Z_v) / 2
    alpha, beta = -245.8909, 261.9373
    gamma1, gamma2, gamma3 = -239.8886, 160.1664, 64.2859
    return alpha + beta * (
        torch.pow(B, gamma1 / 10000)
        * torch.pow(A, gamma2 / 10000)
        * torch.pow(Z, gamma3 / 10000)
    )


def _colorfulness(img):
    import torch

    R, G, Bc = img[0], img[1], img[2]
    apha = R - G
    beta = (R + G) * 0.5 - Bc
    mean_a, mean_b = torch.mean(apha), torch.mean(beta)
    sigma_a, sigma_b = torch.std(apha), torch.std(beta)
    if mean_a == 0 or mean_b == 0 or sigma_a == 0 or sigma_b == 0:
        return torch.tensor(0.0)
    return 0.02 * torch.log(sigma_a ** 2 / torch.abs(mean_a) ** 0.2) * torch.log(
        sigma_b ** 2 / torch.abs(mean_b) ** 0.2
    )


def _img2patch(img, pch_size, stride=1):
    import torch

    pch_H = pch_W = pch_size
    C, H, W = img.shape
    num_H = len(range(0, H - pch_H + 1, stride))
    num_W = len(range(0, W - pch_W + 1, stride))
    num_pch = num_H * num_W
    pch = torch.zeros((C, pch_H * pch_W, num_pch))
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = img[:, ii:H - pch_H + ii + 1:stride, jj:W - pch_W + jj + 1:stride]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1
    return pch.reshape((C, pch_H, pch_W, num_pch))


def _noise(img, pch_size=8):
    import torch

    pch = _img2patch(img, pch_size, 3)
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))
    d = pch.shape[0]
    mu = pch.mean(dim=1, keepdim=True)
    X = pch - mu
    sigma_X = torch.matmul(X, torch.transpose(X, 0, 1)) / num_pch
    try:
        sig_value, _ = torch.linalg.eigh(sigma_X, UPLO="U")
    except RuntimeError:
        return torch.tensor(0.0)
    sig_value.sort()
    for ii in range(-1, -d - 1, -1):
        tau = torch.mean(sig_value[:ii])
        if torch.sum(sig_value[:ii] > tau) == torch.sum(sig_value[:ii] < tau):
            if tau < 0 or torch.isnan(tau):
                return torch.tensor(0.0)
            return torch.sqrt(tau)
    return torch.tensor(0.0)


def _calc_hist(image):
    import cv2

    h, w = image.shape
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]) / (h * w)
    return hist.flatten()


def _sobel_sharpness(img):
    import cv2

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gm = cv2.sqrt(sobelx * sobelx + sobely * sobely)
    return np.mean(gm), gm


def _exposure_wellness(img, miu, denom=0.001):
    m = np.exp(-(img - miu) * (img - miu) / denom)
    return np.mean(m), m


def _exposure_bright_dark(img_hue, img_sat, img_bri, bri_mm, thresh, abs_delta, is_bright, denom=0.01):
    ex_bri, ex_sat, ex_hue = 0, 1, 0
    ex_map, ex_sat_map = None, None
    delta = 1 - thresh if is_bright else -thresh
    if (is_bright and bri_mm > thresh) or ((not is_bright) and bri_mm < thresh):
        ex_bri, ex_map = _exposure_wellness(img_bri, thresh + delta, denom)
    if is_bright:
        ex_sat, ex_sat_map = _exposure_wellness(img_sat, 0, 0.01)
        ex_hue, ex_hue_map = _exposure_wellness(img_hue, 0.002, 0.00001)
        ex_hue *= np.exp(-10 * np.std(ex_hue_map))
    return ex_bri, ex_hue, ex_sat, ex_map, ex_sat_map


def _find_texture_center(smap):
    h, w = smap.shape
    total = np.sum(smap)
    center_s = 0
    stride = 4
    top = bottom = left = right = 0
    for i in range(0, h - stride, stride):
        if np.sum(smap[i:, :]) < 0.995 * total:
            top = i - stride
            center_s = np.sum(smap[top:, :])
            break
    for j in range(0, h - stride, stride):
        if np.sum(smap[top:h - j + 1, :]) < 0.995 * center_s:
            bottom = j - stride
            center_s = np.sum(smap[top:(h - bottom + 1), :])
            break
    for i in range(0, w - stride, stride):
        if np.sum(smap[top:(h - bottom + 1), i:]) < 0.995 * center_s:
            left = i - stride
            center_s = np.sum(smap[top:(h - bottom + 1), left:])
            break
    for j in range(0, w - stride, stride):
        if np.sum(smap[top:(h - bottom + 1), left:(w - j + 1)]) < 0.995 * center_s:
            right = j - stride
            break
    return top, bottom, left, right, smap[top:(h - bottom + 1), left:(w - right + 1)]


def _exposure_frame(input_img, bright_thresh=0.95, dark_thresh=0.1):
    import cv2

    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    h, w, _ = img.shape
    img_bri = img[:, :, 2] / 255.0
    sharp, smap = _sobel_sharpness(img_bri)
    bri_max, bri_min = np.max(img_bri), np.min(img_bri)
    if np.sum(smap) < 0.01:
        return 0, 0
    if bri_max < dark_thresh:
        return 0, 1
    if bri_min > bright_thresh:
        return 1, 0
    top, bottom, left, right, smap = _find_texture_center(smap)
    img = img[top:(h - bottom + 1), left:(w - right + 1), :]
    img_hue, img_sat, img_bri = img[:, :, 0] / 255.0, img[:, :, 1] / 255.0, img[:, :, 2] / 255.0
    bri_mean = np.mean(img_bri)
    hist_sat = _calc_hist(img[:, :, 1])
    over_bri, ex_hue, ex_sat, _, _ = _exposure_bright_dark(
        img_hue, img_sat, img_bri, bri_max, bright_thresh, 0.035, True, denom=0.01
    )
    under_bri, _, _, _, _ = _exposure_bright_dark(
        img_hue, img_sat, img_bri, bri_min, dark_thresh, 0.025, False, denom=0.01
    )
    hue_weight = np.exp(-5 * (0.3 - bri_mean)) if bri_mean < 0.3 else 1
    ret_under = under_bri * np.exp(-5 * sharp)
    sat_sim = 0.5 * np.sum(hist_sat[:1]) + 0.35 * np.sum(hist_sat[:5]) + 0.15 * np.sum(hist_sat[:25])
    sat_sim = 1 - np.exp(-5 * sat_sim)
    ssmr = np.std(smap) / sharp
    ssw = 1 if ssmr < 2 else (0.25 if ssmr > 4 else (1.75 - 0.375 * ssmr))
    ret_over = ex_hue * hue_weight * ex_sat + over_bri * (
        1 - np.exp(-2.5 * (ex_sat + sat_sim))
    ) * np.exp(-0.5 * sharp) * ssw
    return float(np.clip(ret_over, 0, 1)), float(np.clip(ret_under, 0, 1))


def _distortion_features(frames_rgb: List[np.ndarray]):
    """Return an (N, 6) tensor of per-frame distortion metrics on CPU."""
    import cv2
    import torch

    rows = []
    for f in frames_rgb:
        chw = torch.from_numpy(np.ascontiguousarray(np.transpose(f, (2, 0, 1)))).float()
        gray = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)).float()
        rows.append(
            [
                float(_sharpness(gray)),
                float(_noise(chw)),
                float(_blockinessII(gray)),
                0.0,
                0.0,
                float(_colorfulness(chw)),
            ]
        )
    feats = torch.as_tensor(rows, dtype=torch.float32)
    overs, unders = [], []
    for f in frames_rgb:
        h, w, _ = f.shape
        small = cv2.resize(f, (w // 2, h // 2))
        o, u = _exposure_frame(small)
        overs.append(o)
        unders.append(u)
    feats[:, 3] = torch.tensor(overs, dtype=torch.float32)
    feats[:, 4] = torch.tensor(unders, dtype=torch.float32)
    # nan/inf guard on the noise column (upstream check_blockiness_values_2).
    for idx in range(feats.shape[0]):
        if torch.isnan(feats[idx, 1]) or torch.isinf(feats[idx, 1]):
            t_sum, cnt = 0.0, 0
            t = idx
            while t >= 0 and cnt < 2:
                if not (torch.isnan(feats[t, 1]) or torch.isinf(feats[t, 1])):
                    t_sum += float(feats[t, 1])
                    cnt += 1
                t -= 1
            t = idx
            r = 0
            while t < feats.shape[0] and r < 2:
                if not (torch.isnan(feats[t, 1]) or torch.isinf(feats[t, 1])):
                    t_sum += float(feats[t, 1])
                    r += 1
                t += 1
            feats[idx, 1] = t_sum / (cnt + r) if (cnt + r) else -245.8909
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Fusion / quality head — ported verbatim from upstream MDVQA_cvpr.
# ─────────────────────────────────────────────────────────────────────────────
def _build_head():
    import torch
    import torch.nn as nn

    class MDVQA_cvpr(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            input_size = sum([256, 160, 128, 64])  # 608 (after the [:, :, 48:] slice)
            self.semantic_temporal = nn.Sequential(
                nn.Linear(input_size, 512), nn.LeakyReLU(inplace),
                nn.Linear(512, 128), nn.LeakyReLU(inplace),
                nn.Linear(128, 64), nn.LeakyReLU(inplace),
            )
            self.semantic_spatial = nn.Sequential(
                nn.Linear(input_size, 512), nn.LeakyReLU(inplace),
                nn.Linear(512, 128), nn.LeakyReLU(inplace),
                nn.Linear(128, 6), nn.LeakyReLU(inplace),
            )
            self.distortion_spatial = nn.Sequential(
                nn.Linear(6, 5), nn.LeakyReLU(inplace),
            )
            self.distortion_temporal = nn.Sequential(
                nn.Linear(6, 5), nn.LeakyReLU(inplace),
                nn.Linear(5, 4), nn.LeakyReLU(inplace),
            )
            self.motion_fc = nn.Sequential(
                nn.Linear(512, 512), nn.LeakyReLU(inplace),
                nn.Linear(512, 128), nn.LeakyReLU(inplace),
                nn.Linear(128, 64), nn.LeakyReLU(inplace),
            )
            self.spatial_temporal_fc = nn.Sequential(
                nn.Linear(64 * 2 + 6 + 5 + 4, 64), nn.LeakyReLU(inplace),
                nn.Linear(64, 16), nn.LeakyReLU(inplace),
                nn.Linear(16, 1), nn.LeakyReLU(inplace),
            )
            self.clip_pooling_t = nn.Sequential(
                nn.Conv1d(64 + 6 + 5 + 4, 64 + 6 + 5 + 4, kernel_size=8, stride=8, padding=0),
                nn.LeakyReLU(),
            )

        def forward(self, semantic_feature, metric_feature, motion_feature):
            semantic_feature = semantic_feature[:, :, 48:]
            semantic_feature_t = torch.abs(semantic_feature[:, 0::2] - semantic_feature[:, 1::2])
            distortion_feature_t = torch.abs(metric_feature[:, 0::2] - metric_feature[:, 1::2])
            semantic_feature_s = semantic_feature[:, 1::2, :]
            distortion_feature_s = metric_feature[:, 1::2, :]
            semantic_feature_t = self.semantic_temporal(semantic_feature_t)
            distortion_feature_t = self.distortion_temporal(distortion_feature_t)
            semantic_feature_s = self.semantic_spatial(semantic_feature_s)
            distortion_feature_s = self.distortion_spatial(distortion_feature_s)
            feature_fuse = torch.cat(
                (semantic_feature_t, distortion_feature_t, semantic_feature_s, distortion_feature_s),
                dim=2,
            )
            feature_fuse = feature_fuse.permute([0, 2, 1])
            feature_fuse = self.clip_pooling_t(feature_fuse)
            feature_fuse = feature_fuse.permute([0, 2, 1])
            feature_m = self.motion_fc(motion_feature)
            feature_fuse = torch.cat((feature_fuse, feature_m), dim=2)
            output = self.spatial_temporal_fc(feature_fuse)
            return torch.mean(output).cpu()

    return MDVQA_cvpr()


def _build_semantic(device):
    """torchvision efficientnet_v2_s multi-level (stages 2-6) mean extractor."""
    import torch
    import torch.nn as nn
    from torchvision import models

    class _Semantic(nn.Module):
        def __init__(self):
            super().__init__()
            m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # model.features[:-1] -> keep stages 0..6, drop the final 1x1 head conv.
            self.features_extractor = nn.Sequential(*list(m.children())[0][:-1])
            self.features_extractor.eval()
            for p in self.features_extractor.parameters():
                p.requires_grad = False

        def forward(self, x):
            means = []
            for ii, sub in enumerate(self.features_extractor):
                x = sub(x)
                if ii >= 2:
                    means.append(torch.mean(x, dim=[2, 3]))
            return torch.cat(means, dim=1)  # (batch, 656)

    return _Semantic().to(device).eval()


def _build_motion(device):
    """torchvision r2plus1d_18 avg-pool (512-d) feature extractor."""
    import torch
    import torch.nn as nn
    from torchvision import models

    class _Motion(nn.Module):
        def __init__(self):
            super().__init__()
            m = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
            self.stem = m.stem
            self.layer1, self.layer2 = m.layer1, m.layer2
            self.layer3, self.layer4 = m.layer3, m.layer4
            self.avgpool = m.avgpool
            self.eval()

        def forward(self, x):  # x: (1, 3, T, H, W)
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return x.flatten(1)  # (1, 512)

    return _Motion().to(device).eval()


class MDVQAModule(PipelineModule):
    name = "mdvqa"
    description = "MD-VQA multi-dimensional UGC live VQA (CVPR 2023; real model only, disabled if unavailable)"
    default_config = {
        "clip_len": 16,     # frames per motion clip (upstream fixed at 16)
        "max_clips": 8,     # cap on sampled clips (bounds compute); N = clips * 16
        "device": "auto",
    }
    metric_groups = {
        "mdvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.clip_len = int(self.config.get("clip_len", 16))
        self.max_clips = int(self.config.get("max_clips", 8))
        self._ml_available = False
        self._backend = None
        self._device = "cpu"
        self._semantic = None
        self._motion = None
        self._head = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch  # noqa: F401
            from huggingface_hub import hf_hub_download

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._semantic = _build_semantic(self._device)
            self._motion = _build_motion(self._device)
            head = _build_head()
            weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_HEAD)
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
            # strict=True: the 32-tensor checkpoint is exactly the fusion/head.
            head.load_state_dict(sd, strict=True)
            self._head = head.to(self._device).eval()
            self._backend = "real"
            self._ml_available = True
            logger.info("MD-VQA loaded real fusion/head (%s) on %s", _HF_HEAD, self._device)
            return
        except Exception as e:  # missing dep, missing weights, or key mismatch
            logger.warning(
                "MD-VQA: real model unavailable (%s: %s); mdvqa_* left unset.",
                type(e).__name__,
                e,
            )
        self._backend = "unavailable"
        self._ml_available = False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "real" or self._head is None:
            return sample  # no trained weights -> do not fabricate
        if not sample.is_video:
            return sample  # motion branch requires a video
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            score = self._score_video(sample)
            if score is not None:
                sample.quality_metrics.mdvqa_score = float(score)
        except Exception as e:
            logger.warning("MD-VQA failed on %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _score_video(self, sample: Sample) -> Optional[float]:
        import torch

        frames = self._decode_frames(sample)
        if len(frames) < self.clip_len:
            return None
        n_clips = min(self.max_clips, len(frames) // self.clip_len)
        if n_clips < 1:
            return None
        n = n_clips * self.clip_len  # multiple of clip_len, even, >= 16
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        sampled = [frames[i] for i in idx]

        semantic = self._semantic_feature(sampled)              # (n, 656) on device
        motion = self._motion_feature(sampled)                  # (n_clips, 512) on device
        metric = _distortion_features(sampled).to(self._device)  # (n, 6)

        with torch.no_grad():
            out = self._head(
                semantic.unsqueeze(0), metric.unsqueeze(0), motion.unsqueeze(0)
            )
        score_ori = float(out.item())
        # Upstream affine calibration to a 0-100 MOS, then normalise to 0-1.
        mos = max(0.0, min(99.99, score_ori * 0.8313 + 21.0112))
        return mos / 100.0

    def _normalize_clip(self, frames_rgb: List[np.ndarray]):
        """(len, 3, H, W) ImageNet-normalised float tensor from RGB uint8 frames."""
        import torch

        mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_STD).view(1, 3, 1, 1)
        arr = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0  # (L, H, W, 3)
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
        return (t - mean) / std

    def _semantic_feature(self, frames_rgb: List[np.ndarray]):
        """Per-frame 656-d EfficientNetV2 multi-level feature (batched by 2)."""
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(frames_rgb), 2):
                batch = self._normalize_clip(frames_rgb[i:i + 2]).to(self._device)
                feats.append(self._semantic(batch))
        return torch.cat(feats, dim=0)

    def _motion_feature(self, frames_rgb: List[np.ndarray]):
        """Per-clip 512-d R(2+1)D feature; each clip resized short-side to 224."""
        import torch
        import torchvision

        feats = []
        with torch.no_grad():
            for k in range(len(frames_rgb) // self.clip_len):
                clip = self._normalize_clip(
                    frames_rgb[k * self.clip_len:(k + 1) * self.clip_len]
                )
                clip = torchvision.transforms.functional.resize(clip, size=224, antialias=True)
                clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self._device)  # (1,3,T,H,W)
                feats.append(self._motion(clip))
        return torch.cat(feats, dim=0)

    def _decode_frames(self, sample: Sample) -> List[np.ndarray]:
        """Decode a bounded, contiguous run of RGB frames."""
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap_frames = self.max_clips * self.clip_len * 4  # oversample, then linspace-subsample
        limit = min(total, cap_frames) if total > 0 else cap_frames
        frames: List[np.ndarray] = []
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
