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
from time import time
from tqdm import tqdm


def extract_3dmm(filevideo, fileboxes, device, opt, verbose=True):
    from ayase.third_party.idreveal.grip_unina.util_read import MockFileBoxes
    from .util_3dmm import Compute3DMMtracked
    from ayase.third_party.idreveal.grip_unina.util_read import ReadingResampledVideo

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    op3 = Compute3DMMtracked(device, opt['resources_path'], return_frame=False)

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps.')

        ops = [video, op2.reset(), op3.reset(), ]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(total=len(video))

        dict_out = dict()
        count = 0
        while True:
            try:
                out = count
                for index_op in range(len(ops)):
                    tic = time()
                    out = ops[index_op](out)
                    toc = time()
                    #if verbose and (count == 0):
                    #    print(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
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
            if verbose:
                pbar.update(1)

    if verbose:
        print(f"total time: {list_times} sec")

    return dict_out


def extract_feats_idreavel(file3dmm, device, opt, list_good_track=None, verbose=True):
    from ayase.third_party.idreveal.grip_unina.util_read import MockFileBoxes
    from ayase.third_party.idreveal.grip_unina.util_read import FilterValues
    from ayase.third_party.idreveal.grip_unina.util_dist import ComputeTemporal
    from .util_idreavel import ComputeIdReveal
    from ayase.third_party.idreveal.grip_unina.util_dist import PassIdentity

    if list_good_track is None:
        op2 = PassIdentity()
    else:
        op2 = FilterValues(condition=lambda x: x in list_good_track,
                       list_data=['boxes', 'points', 'image_inds', 'id_track', '3dmm'],
                       key_values='id_track')
    id_reavel = ComputeIdReveal(opt['model']['clip_length'], device, os.path.join(opt['resources_path'], 'model_idreveal.th'))
    op3 = ComputeTemporal(opt['model']['clip_length'], opt['model']['clip_ref_stride'], {'3dmm': id_reavel})

    with MockFileBoxes(file3dmm, list_data=['boxes', 'points', 'image_inds', 'id_track', '3dmm']) as data:
        ops = [data, op2.reset(), op3.reset(), ]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(range(len(data)), total=len(data))
        else:
            pbar = range(len(data))

        dict_out = {'embs_track': list()}
        for count in pbar:
            out = count
            for index_op in range(len(ops)):
                tic = time()
                out = ops[index_op](out)
                toc = time()
                #if verbose and (count == 0):
                #    print(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
                list_times[index_op] += toc - tic

            for key in out.keys():
                if len(out[key]) == 0:
                    continue
                if key in dict_out:
                    dict_out[key].extend(list(out[key]))
                else:
                    dict_out[key] = list(out[key])
            if verbose:
                pbar.set_description('%d ' % len(dict_out['embs_track']))

    if verbose:
        print(f"total time: {list_times} sec")

    return dict_out
