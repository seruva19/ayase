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
import numpy as np
from ayase.third_party.idreveal.grip_unina.util_read import ReadingResampledVideo
from ayase.third_party.idreveal.grip_unina.util_face import DetectFace, ComputeTrack
from ayase.third_party.idreveal.grip_unina.util_dist import ComputeParallelDict, ComputeDistance, ComputeTemporal
from .util_3dmm import Compute3DMMtracked
from .util_idreavel import ComputeIdReveal


class IdReveal:
    def __init__(self, logger, poi_folders, opt, device):
        import yaml
        self.logger = logger
        self.device = device
        self.opt = opt
        self.list_poi = list(poi_folders.keys())
        self.featname = '3dmm'

        # duration_sec = final_mean*time/ fps
        for l in yaml.dump(self.opt).splitlines():
            self.logger.info("  " + l)

        self.op2 = DetectFace(self.device, os.path.join(self.opt['resources_path'], 'Resnet50_Final.pth'),
                              size_threshold=self.opt['face_det']['size_threshold'], batch_size=self.opt['rec_stride'],
                              score_threshold=self.opt['face_det']['score_threshold'])
        self.op3 = ComputeTrack(self.opt['face_det']['iou_threshold'])
        self.op4 = Compute3DMMtracked(self.device,
                                      self.opt['resources_path'], return_frame=False)
        if self.op4 is None:
            self.logger.error(f"Feature {self.featname} not found!")
            assert False

        clip_op = ComputeIdReveal(self.opt['model']['clip_length'], device,
                                  os.path.join(self.opt['resources_path'], 'model_idreveal.th'))
        self.op5 = ComputeTemporal(self.opt['model']['clip_length'],
                                   self.opt['model']['clip_stride'],
                                   {self.featname: clip_op})

        dict_ops = {_: ComputeDistance(self.featname, poi_folders[_],
                                       normalize=self.opt['dist_normalization'])
                    for _ in self.list_poi}
        self.op6 = ComputeParallelDict(['embs_'+self.featname, ], 'embs_dists', dict_ops)

        self.logger.info(f"Created the network on {self.device}")

    def compute_distance_video(self, filevideo, list_poi=None, verbose=True):
        from time import time
        from tqdm import tqdm
        with ReadingResampledVideo(filevideo, self.opt['fps'], self.opt['read_stride']) as video:
            self.logger.info(f'Reading video {filevideo} of {video.get_number_frames()} '
                             f'frames with {video.get_fps()} fps.')

            ops = [video,
                   self.op2.reset(), self.op3.reset(),
                   self.op4.reset(), self.op5.reset(),
                   self.op6.reset(list_name=list_poi, filename=filevideo)]
            list_times = [0 for _ in range(len(ops))]
            pbar = tqdm(total=len(video), disable=not verbose)

            dict_out = {'embs_track': list()}
            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        tic = time()
                        out = ops[index_op](out)
                        toc = time()
                        if count == 0:
                            self.logger.debug(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1
                # if count==3: break
                for key in out.keys():
                    if len(out[key]) == 0:
                        continue
                    if key in dict_out:
                        dict_out[key].extend(list(out[key]))
                    else:
                        dict_out[key] = list(out[key])

                pbar.update(1)
                pbar.set_description('%d %d' % (self.op3.num_tracks(), len(dict_out['embs_track'])))
                if not verbose:
                    self.logger.debug('pbar %d/%d: %d %d ' % (count, len(video),
                                            self.op3.num_tracks(), len(dict_out['embs_track'])))

        self.logger.info(f"total time: {list_times} sec")
        info_tracks = self.op3.info_tracks()
        return dict_out, info_tracks

    def compute_distance(self, filevideo, list_poi=None, verbose=True):
        return self.compute_distance_video(filevideo, list_poi=list_poi, verbose=verbose)

    def merge_track(self, dists, rangs):
        if self.opt['final_mean'] > 1:
            if len(dists) < self.opt['final_mean']:
                return list(), list()
            half = self.opt['final_mean'] // 2
            from scipy.ndimage import uniform_filter
            if len(dists[0].shape) > 0:
                dists = np.nanmax(dists, -1)

            dists = uniform_filter(dists, self.opt['final_mean'])[half:-half]
            rangs = rangs[half:-half]

        if self.opt['percentile'] >= 0:
            dists = [np.percentile(dists, self.opt['percentile']), ]
            loc = [(np.min(rangs), np.max(rangs)), ]
        else:
            loc = rangs

        return dists, loc
