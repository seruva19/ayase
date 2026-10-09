"""S3FD face detector (port of ``detectors/s3fd`` from joonson/syncnet_python, MIT)."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from .box_utils import nms_
from .nets import S3FDNet

# Per-channel mean subtracted when the detector was trained.
_IMG_MEAN = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")
_FINAL_NMS_THRESH = 0.1


class S3FD:
    """S3FD with loaded weights."""

    def __init__(self, weights_path: Path, device: torch.device) -> None:
        self.device = device
        self.net = S3FDNet().to(self.device)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def detect_faces(
        self, image: np.ndarray, conf_th: float = 0.8, scales: Tuple[float, ...] = (1.0,)
    ) -> np.ndarray:
        """Detect faces on one RGB ``uint8`` frame.

        Returns ``(N, 5)`` rows ``x1, y1, x2, y2, score`` in pixels of the input frame.
        """
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR
                )

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype("float32") - _IMG_MEAN
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                detections = self.net(x)
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while j < detections.size(2) and detections[0, i, j, 0] > conf_th:
                        score = float(detections[0, i, j, 0])
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bboxes = np.vstack((bboxes, (pt[0], pt[1], pt[2], pt[3], score)))
                        j += 1

        keep = nms_(bboxes, _FINAL_NMS_THRESH)
        return bboxes[keep]
