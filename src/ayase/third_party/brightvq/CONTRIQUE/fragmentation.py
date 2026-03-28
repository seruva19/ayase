import os
import os.path as osp
import random
from functools import lru_cache
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import pandas as pd
import sys 

sys.path.append("/work/09032/saini_2/ls6/LIVE-Work/UGC-HDR/QA-models/")
from util_hdr_10bit import *
random.seed(42)


def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    upsample=-1,
    **kwargs,
):
    if upsample > 0:
        old_h, old_w = video.shape[-2], video.shape[-1]
        if old_h >= old_w:
            w = upsample
            h = int(upsample * old_h / old_w)
        else:
            h = upsample
            w = int(upsample * old_w / old_h)
        
        video = get_resized_video(video, h, w)
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:

        ovideo = video
        video = torch.nn.functional.interpolate(
            video, scale_factor=1 / ratio, mode="bilinear"
        )
        
    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video, scale_factor=randratio, mode="bilinear"
        )
        
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))

def get_resized_video(
    video, size_h=224, size_w=224, random_crop=False, arp=False, **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


def get_single_view(
    video, sample_type="aesthetic", **kwargs,
):
    if sample_type.startswith("aesthetic"):
        video = get_resized_video(video, **kwargs)
    elif sample_type.startswith("technical"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type.startswith("semantic"):
        video = get_resized_video(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def spatial_temporal_view_decomposition(
    frames, sample_types, samplers, is_train=False, 
):
    video = {}
    for stype in samplers:
        frame_inds = samplers[stype](len(frames), is_train)
        imgs = [torch.from_numpy(frames[idx]) for idx in frame_inds]
        video[stype] = torch.stack(imgs, 0).permute(0, 3, 1, 2)
    del frames
    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)
    return sampled_video, frame_inds


class UnifiedFrameSampler:
    def __init__(
        self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0.0,
    ):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):

        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )

        drop = random.sample(
            list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate)
        )
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


def ViewDecompositionDataset(opt, frames):
    sample_types = opt["sample_types"]
    mean = torch.FloatTensor([0.4527, 0.4374, 0.4202])
    std = torch.FloatTensor([0.1807,0.1838, 0.1826])
    mean_semantic = torch.FloatTensor([122.77, 116.75, 104.09])/255.0
    std_semantic = torch.FloatTensor([68.50, 66.63, 70.32])/255.0
    samplers = {}
    for stype, sopt in opt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in COVER
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in COVER
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
        
    try:
        ## Read Original Frames
        ## Process Frames
        data, frame_inds = spatial_temporal_view_decomposition(
            frames,
            sample_types,
            samplers,
            opt
        )
        """
        for k, v in data.items():
            if k == "technical" or k == "aesthetic":
                data[k] = ((v.permute(1, 2, 3, 0) - mean) / std).permute(
                    3, 0, 1, 2
                )
            elif k == "semantic":
                data[k] = ((v.permute(1, 2, 3, 0) - mean_semantic) / std_semantic).permute(3, 0, 1, 2)
        """ 
        data["frame_inds"] = frame_inds
    except Exception as e:
        # exception flow
        raise ValueError("Error in Spatio-temporal sampling: ",e)
        return None
    
    return data
