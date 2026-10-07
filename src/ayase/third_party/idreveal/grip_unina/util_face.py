#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import os
import torch
import numpy as np
from .util_read import BGR2RGBs
from scipy.optimize import linear_sum_assignment


class DetectFace:
    def __init__(self, device, weights=None, size_threshold=75, target_size=1280,
                 batch_size=16, score_threshold=0.7, iou_threshold=0.5, return_frame=True):
        from ayase.third_party.idreveal.retinaface import get_detector
        self.device = device
        self.batch_size = batch_size
        self.retinaface = get_detector(network='resnet50', device=self.device, trained_model=weights).eval()
        self.target_size = target_size
        self.size_threshold = size_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.return_frame = return_frame

    def detect_faces(self, images_torch):
        from ayase.third_party.idreveal.retinaface import detect_video_torch
        resize_factor = min(self.target_size / max(images_torch.shape[-1], images_torch.shape[-2]), 1.0)
        with torch.no_grad():
            return detect_video_torch(self.retinaface, images_torch, resize_factor=resize_factor,
                                      score_threshold=self.score_threshold, iou_threshold=self.iou_threshold,
                                      size_threshold=self.size_threshold, batch_size=self.batch_size,
                                      resize_image=resize_factor != 1.0)

    def reset(self):
        return self

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            imgs = [inp['frame_bgr'], ]
            ids = [inp['image_ind'], ]
            del inp['image_ind']
            if not self.return_frame:
                del inp['frame_bgr']
        else:
            imgs = inp['frames_bgr']
            ids = inp['frames_inds']
            if not self.return_frame:
                del inp['frames_bgr']

        imgs = BGR2RGBs(imgs)
        images_torch = (torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(self.device) - 128) / 128.0

        # face detector
        image_inds, scores, boxes, points = self.detect_faces(images_torch)
        del images_torch
        image_inds = np.asarray([ids[int(x)] for x in image_inds.cpu().numpy()])
        order = np.argsort(image_inds)
        inp['boxes'] = list(boxes.cpu().numpy()[order])
        inp['boxes_score'] = list(scores.cpu().numpy()[order])
        inp['points'] = list(points.cpu().numpy()[order])
        inp['image_inds'] = image_inds[order]

        return inp


def iou(boxes0, boxes1):
    a0 = (boxes0[..., 2] - boxes0[..., 0]) * (boxes0[..., 3] - boxes0[..., 1])
    a1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    w = np.minimum(boxes0[..., 2], boxes1[..., 2]) - np.maximum(boxes0[..., 0], boxes1[..., 0])
    h = np.minimum(boxes0[..., 3], boxes1[..., 3]) - np.maximum(boxes0[..., 1], boxes1[..., 1])
    wh = np.maximum(0.0, w) * np.maximum(0.0, h)
    return wh / (a0 + a1 - wh)


class ComputeTrack:

    def __init__(self, thres=0.5):
        self.lst_boxes = list()
        self.lst_track = list()
        self.thres = thres
        self.detected_tracks = 0
        self.detected_infos = list()

    def reset(self):
        self.lst_boxes = list()
        self.lst_track = list()
        self.detected_tracks = 0
        self.detected_infos = list()
        return self

    def num_tracks(self):
        return self.detected_tracks

    def info_tracks(self):
        return self.detected_infos

    def single_frame(self, frame_ind, new_boxes):

        if len(new_boxes) == 0:
            self.lst_track, self.lst_boxes = list(), list()
            return list()

        new_track = [-1, ] * len(new_boxes)

        if len(self.lst_boxes) == 0:
            assigned_rows = list()
            assigned_cols = list()
            new_cols = list(range(len(new_boxes)))
            ass_scores = [-1, ] * len(new_boxes)
        else:
            scores = iou(np.asarray(new_boxes)[None, :, :], np.asarray(self.lst_boxes)[:, None, :])
            assigned_rows, assigned_cols = linear_sum_assignment(1.0-scores)
            ass_scores = [-1,] * len(new_boxes)
            for a, b in zip(assigned_rows, assigned_cols):
                ass_scores[b] = scores[a, b]
            val = [(scores[a, b] > self.thres) for a, b in zip(assigned_rows, assigned_cols)]
            assigned_rows = [a for a, b in zip(assigned_rows, val) if b]
            assigned_cols = [a for a, b in zip(assigned_cols, val) if b]

            if len(assigned_cols) < len(new_boxes):
                new_cols = [a for a in range(len(new_boxes)) if a not in assigned_cols]
                print(len(new_boxes), scores[:, new_cols])
            else:
                new_cols = list()

        for a, b in zip(assigned_rows, assigned_cols):
            new_track[b] = self.lst_track[a]
            self.detected_infos[new_track[b]][-1] += 1

        for b in new_cols:
            new_track[b] = self.detected_tracks
            self.detected_infos.append([frame_ind, ass_scores[b], 1])
            self.detected_tracks = self.detected_tracks + 1

        self.lst_boxes = new_boxes
        self.lst_track = new_track

        return list(new_track)

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            track = self.single_frame(inp['image_ind'], inp['boxes'])
        else:
            num_boxes = len(inp['boxes'])
            track = [-1, ] * num_boxes
            for frame_ind in inp['frames_inds']:
                pos_boxes1 = [x for x in range(num_boxes) if inp['image_inds'][x] == frame_ind]
                new_boxes1 = [inp['boxes'][x] for x in pos_boxes1]
                track1 = self.single_frame(frame_ind, new_boxes1)
                for i, x in enumerate(pos_boxes1):
                    track[x] = track1[i]

        inp['id_track'] = track
        return inp


class ComputeLandMarkers:

    def __init__(self, device):
        import face_alignment

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

    def reset(self):
        return self

    def __call__(self, inp):
        # ['face_bgr', 'face_start', 'points', 'boxes']
        if 'face_bgr' in inp:
            imgs = inp['face_bgr']
            face_start = inp['face_start']

        else:
            imgs = inp['frames_bgr']
            image_inds = inp['frames_inds']
            image_inds = [image_inds.index(x) for x in inp['image_inds']]
            imgs = [imgs[x] for x in image_inds]
            del image_inds
            face_start = [(0.0, 0.0), ] * len(imgs)

        boxes = inp['boxes']
        inp['landmarks68'] = list()

        for index in range(len(imgs)):
            img = imgs[index][:, :, ::-1]
            ss = face_start[index]
            box = np.asarray(boxes[index]) - [ss[0], ss[1], ss[0], ss[1]]

            preds = self.fa.get_landmarks(img, detected_faces=[box, ])[0]
            preds = preds + [[ss[0], ss[1]], ]
            inp['landmarks68'].append(preds)

        return inp

