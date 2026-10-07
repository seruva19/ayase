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


import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ayase.third_party.idreveal.TDDFA import TDDFA, get_mb1

class Compute3DMMtracked:

    def __init__(self, device, dir_weights=None, return_frame=False):
        cfg = get_mb1(dir_weights)
        self.device = device
        self.tddfa = TDDFA(device=self.device, **cfg)
        self.return_frame = return_frame

    def reset(self):
        return self

    def __call__(self, inp):
        assert 'frames_inds' in inp
        imgs = self.tddfa.frames2torch(inp['frames_bgr'])
        frames_inds = inp['frames_inds']
        image_inds = list(inp['image_inds'])

        if not self.return_frame:
            del inp['frames_bgr']
            del inp['frames_inds']

        num_boxes = len(image_inds)
        if num_boxes == 0:
            inp['3dmm'] = list()
            inp['roibox'] = list()
            return inp
        image_inds_local = [frames_inds.index(x) for x in image_inds]
        with torch.no_grad():
            new_roibox = self.tddfa.parse_roi_box_from_bboxes(inp['boxes'])
            new_param  = self.tddfa.get_3dmm_from_crops(self.tddfa.crop_imgs(imgs, image_inds_local, new_roibox))
            new_landmark = self.tddfa.recons_sparse(new_param, new_roibox)
            new_roibox = self.tddfa.parse_roi_box_from_landmarkes(new_landmark)
            new_param = self.tddfa.get_3dmm_from_crops(self.tddfa.crop_imgs(imgs, image_inds_local, new_roibox))

        inp['3dmm'] = new_param
        inp['roibox'] = new_roibox
        return inp


class Compute3DMMtracking:

    def __init__(self, device, dir_weights=None, return_frame=False, onyl_track=False, recompute_feats=True, th=2448):
        cfg = get_mb1(dir_weights)
        self.device = device
        self.tddfa = TDDFA(device=self.device, **cfg)
        self.lst_landmark = list()
        self.lst_track = list()
        self.dense_flag = False
        self.return_frame = return_frame
        self.onyl_track = onyl_track
        self.detected_tracks = 0
        self.detected_infos = list()
        self.recompute_feats = recompute_feats
        self.th = th
        if not self.recompute_feats:
            self.th = 2*th

    def num_tracks(self):
        return self.detected_tracks

    def info_tracks(self):
        return self.detected_infos

    def boxes2landmark(self, frame_bgr, boxes):
        with torch.no_grad():
            return list(self.tddfa.recon_vers(*self.tddfa(frame_bgr, boxes), dense_flag=self.dense_flag))

    def param2landmark(self, param, roibox):
        with torch.no_grad():
            if len(param) == 0:
                return list()
            else:
                return list(self.tddfa.recon_vers(param, roibox, dense_flag=self.dense_flag))

    def landmark2param(self, frame_bgr, landmark):
        with torch.no_grad():
            return self.tddfa(frame_bgr, landmark, crop_policy='landmark')

    def reset(self):
        self.lst_landmark = list()
        self.lst_track = list()
        self.detected_tracks = 0
        self.detected_infos = list()
        return self

    def single_frame(self, frame_ind, frame_bgr, new_landmark):
        num_faces = len(new_landmark)
        if num_faces == 0:
            self.lst_track, self.lst_landmark = list(), list()
            return list(), list(), list()

        lst_param = list()
        lst_roibox = list()
        if self.recompute_feats and (len(self.lst_landmark) > 0):
            lst_roibox = self.tddfa.parse_roi_box_from_landmarkes(self.lst_landmark)
            lst_param = self.tddfa.get_3dmm_from_crops(self.tddfa.crop_single_img(frame_bgr, lst_roibox))

            lst_val = [(abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1])) >= 2020 for roi_box in lst_roibox]
            lst_param  = [a for a, b in zip(lst_param, lst_val) if b]
            lst_roibox = [a for a, b in zip(lst_roibox, lst_val) if b]
            self.lst_track = [a for a, b in zip(self.lst_track, lst_val) if b]
            self.lst_landmark = self.tddfa.recons_sparse(lst_param, lst_roibox)
            del lst_val

        if len(self.lst_landmark) == 0:
            assigned_rows = list()
            assigned_cols = list()
            new_cols = list(range(num_faces))
            ass_dists = [-1 for _ in range(num_faces)]
        else:
            dists = np.sum(np.square(
                np.asarray(new_landmark)[None, :, :, :] - np.asarray(self.lst_landmark)[:, None, :, :]),
                           (-1, -2))
            assigned_rows, assigned_cols = linear_sum_assignment(dists)
            ass_dists = [-1 for _ in range(num_faces)]
            for a, b in zip(assigned_rows, assigned_cols):
                ass_dists[b] = dists[a, b]
            val = [(dists[a, b] < self.th) for a, b in zip(assigned_rows, assigned_cols)]
            assigned_rows = [a for a, b in zip(assigned_rows, val) if b]
            assigned_cols = [a for a, b in zip(assigned_cols, val) if b]
            new_cols = [a for a in range(num_faces) if a not in assigned_cols]
            if len(assigned_cols) < num_faces:
                print(dists[:, new_cols])

        new_track = [-1, ] * num_faces
        new_param = [None, ] * num_faces
        new_roibox = [None, ] * num_faces
        new_landmark = [new_landmark[b] for b in range(num_faces)]
        for a, b in zip(assigned_rows, assigned_cols):
            new_track[b] = self.lst_track[a]
            self.detected_infos[new_track[b]][-1] += 1
            if self.recompute_feats:
                new_landmark[b] = self.lst_landmark[a]
                new_param[b] = lst_param[a]
                new_roibox[b] = lst_roibox[a]

        for b in new_cols:
            new_track[b] = self.detected_tracks
            self.detected_infos.append([frame_ind, ass_dists[b], frame_ind+1])
            self.detected_tracks = self.detected_tracks + 1

        self.lst_landmark = new_landmark
        self.lst_track = new_track

        return list(new_track), list(new_param), list(new_roibox)

    def __call__(self, inp):
        assert 'frames_inds' in inp
        imgs = self.tddfa.frames2torch(inp['frames_bgr'])
        frames_inds = inp['frames_inds']
        image_inds = list(inp['image_inds'])

        if not self.return_frame:
            del inp['frames_bgr']
            del inp['frames_inds']

        num_boxes = len(image_inds)
        if num_boxes == 0:
            self.lst_track, self.lst_landmark = list(), list()
            inp['id_track'] = list()
            if not self.onyl_track:
                inp['3dmm'] = list()
                inp['roibox'] = list()
            return inp
        image_inds_local = [frames_inds.index(x) for x in image_inds]
        new_roibox = self.tddfa.parse_roi_box_from_bboxes(inp['boxes'])
        new_param = self.tddfa.get_3dmm_from_crops(self.tddfa.crop_imgs(imgs, image_inds_local, new_roibox))
        new_landmark = self.tddfa.recons_sparse(new_param, new_roibox)
        new_roibox = self.tddfa.parse_roi_box_from_landmarkes(new_landmark)
        new_param = self.tddfa.get_3dmm_from_crops(self.tddfa.crop_imgs(imgs, image_inds_local, new_roibox))
        new_landmark = self.tddfa.recons_sparse(new_param, new_roibox)
        new_track = [-1, ] * num_boxes
        new_ind_val = [i for i in range(num_boxes)
                       if (abs(new_roibox[i][2] - new_roibox[i][0]) * abs(new_roibox[i][3] - new_roibox[i][1])) >= 2020]

        for frame_ind, frame_bgr in zip(frames_inds, imgs):
            pos_boxes1 = [x for x in new_ind_val if image_inds[x] == frame_ind]
            new_landmark1 = [new_landmark[x] for x in pos_boxes1]
            track1, param1, roibox1 = self.single_frame(frame_ind, frame_bgr, new_landmark1)
            for i, x in enumerate(pos_boxes1):
                new_track[x] = track1[i]
                if param1[i] is not None:
                    new_param[x] = param1[i]
                    new_roibox[x] = roibox1[i]

        inp['id_track'] = new_track
        if not self.onyl_track:
            inp['3dmm'] = new_param
            inp['roibox'] = new_roibox
        return inp
