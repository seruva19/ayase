"""S3FD box helpers (port of ``detectors/s3fd/box_utils.py`` from syncnet_python, MIT)."""

# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see PY_FASTER_RCNN_LICENSE.md for details]
# Written by Ross Girshick

from itertools import product
from typing import List, Tuple

import numpy as np
import torch

DETECT_TOP_K = 750
DETECT_NMS_THRESH = 0.3
DETECT_CONF_THRESH = 0.05
DETECT_NMS_TOP_K = 5000
PRIOR_VARIANCE = (0.1, 0.2)
PRIOR_MIN_SIZES = (16, 32, 64, 128, 256, 512)
PRIOR_STEPS = (4, 8, 16, 32, 64, 128)


def nms_(dets: np.ndarray, thresh: float) -> np.ndarray:
    """Non-maximum suppression on numpy detections ``(N, 5)``: ``x1, y1, x2, y2, score``.

    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(np.intp)


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: Tuple[float, float]) -> torch.Tensor:
    """Decode predicted offsets ``(num_priors, 4)`` against centre-size priors into boxes."""
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, overlap: float = 0.5, top_k: int = 200
) -> Tuple[torch.Tensor, int]:
    """Non-maximum suppression on torch tensors; returns ``(kept indices, count)``."""
    keep = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    _, idx = scores.sort(0)
    idx = idx[-top_k:]

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        xx1 = torch.clamp(torch.index_select(x1, 0, idx), min=x1[i])
        yy1 = torch.clamp(torch.index_select(y1, 0, idx), min=y1[i])
        xx2 = torch.clamp(torch.index_select(x2, 0, idx), max=x2[i])
        yy2 = torch.clamp(torch.index_select(y2, 0, idx), max=y2[i])
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]
    return keep, count


class Detect:
    """Post-processing of S3FD outputs: box decoding and NMS."""

    def __init__(
        self,
        num_classes: int = 2,
        top_k: int = DETECT_TOP_K,
        nms_thresh: float = DETECT_NMS_THRESH,
        conf_thresh: float = DETECT_CONF_THRESH,
        variance: Tuple[float, float] = PRIOR_VARIANCE,
        nms_top_k: int = DETECT_NMS_TOP_K,
    ) -> None:
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(
        self, loc_data: torch.Tensor, conf_data: torch.Tensor, prior_data: torch.Tensor
    ) -> torch.Tensor:
        """Return detections ``(batch, num_classes, top_k, 5)``: ``score, x1, y1, x2, y2``."""
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = min(count, self.top_k)

                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1
                )

        return output


class PriorBox:
    """S3FD prior boxes for an input size and its feature-map sizes."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        feature_maps: List[List[int]],
        min_sizes: Tuple[int, ...] = PRIOR_MIN_SIZES,
        steps: Tuple[int, ...] = PRIOR_STEPS,
    ) -> None:
        self.imh = input_size[0]
        self.imw = input_size[1]
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.steps = steps

    def forward(self) -> torch.Tensor:
        """Return priors ``(num_priors, 4)`` as ``cx, cy, w, h`` in image fractions."""
        mean: List[float] = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        return torch.FloatTensor(mean).view(-1, 4)
