# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn
from torchvision.ops import roi_align

from . import models
from .utils.io import _load
from .utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from .utils.tddfa_util import (
    load_model,
    ToTensorGjz, NormalizeGjz,
    recon_dense, recon_sparse
)


make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)




class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        torch.set_grad_enabled(False)

        # config
        self.device = kvs.get('device', 'cpu')
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, kvs.get('checkpoint_fp'))

        if self.device != 'cpu':
            cudnn.benchmark = True
        model = model.to(self.device)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = torch.from_numpy(r.get('mean')[None, :]).float().to(self.device)
        self.param_std = torch.from_numpy(r.get('std')[None, :]).float().to(self.device)

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def frames2torch(self, frames):
        return torch.stack([self.transform(img) for img in frames], 0)

    def crop_single_img(self, image_torch, boxes):
        if not torch.is_tensor(boxes):
            boxes = torch.from_numpy(np.asarray(boxes))

        boxes = boxes.float() + torch.tensor([-0.5, -0.5, -0.5, -0.5], device=boxes.device, dtype=torch.float32)
        b = torch.cat((torch.zeros((len(boxes), 1), dtype=boxes.dtype, device=boxes.device),
                       boxes), -1).to(image_torch.device)
        return roi_align(image_torch[None, ...], b, (self.size, self.size))

    def crop_imgs(self, images_torch, image_inds, boxes):
        if not torch.is_tensor(boxes):
            boxes = torch.from_numpy(np.asarray(boxes))
        if not torch.is_tensor(image_inds):
            image_inds = torch.from_numpy(np.asarray(image_inds))

        boxes = boxes.float() + torch.tensor([-0.5, -0.5, -0.5, -0.5], device=boxes.device, dtype=torch.float32)
        b = torch.cat((image_inds[:, None].float(), boxes), -1).to(images_torch.device)
        return roi_align(images_torch, b, (self.size, self.size))

    def get_3dmm_from_crops(self, crops_torch):
        with torch.no_grad():
            param = self.model(crops_torch.to(self.device))
            param = param * self.param_std + self.param_mean  # re-scale
            param = param.cpu().numpy().astype(np.float32)
        return param

    def parse_roi_box_from_bboxes(self, boxes):
        return np.stack([parse_roi_box_from_bbox(obj) for obj in boxes], 0)

    def parse_roi_box_from_landmarkes(self, landmark):
        return np.stack([parse_roi_box_from_landmark(obj) for obj in landmark], 0)

    def recons_sparse(self, param_lst, roi_box_lst):
        ver_lst = [recon_sparse(param, roi_box, self.size)
         for param, roi_box in zip(param_lst, roi_box_lst)]

        return ver_lst
