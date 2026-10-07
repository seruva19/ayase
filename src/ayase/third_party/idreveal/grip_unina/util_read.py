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
import cv2


class ReadingVideo:
    
    def __init__(self, filename, stride=0):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.length = 0
        self.count = 0
        
    def get_number_frames(self):
        return self.length
    
    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        
    def get_fps(self):
        return self.video_cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self):
        self.count = 0
        self.video_cap = cv2.VideoCapture(self.filename)
        self.length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stride <= 0:
            ret, img = self.video_cap.read()
            if not ret:
                raise StopIteration
            image_ind = self.count
            self.count = self.count+1
            return {'frame_bgr': img, 'image_ind': image_ind}
        else:
            out = {'frames_bgr': list(), 'frames_inds': list()}
            for _ in range(self.stride):
                ret, img = self.video_cap.read()
                if not ret:
                    break
                out['frames_bgr'].append(img)
                out['frames_inds'].append(self.count)
                self.count = self.count+1
            
            if len(out['frames_bgr']) == 0:
                raise StopIteration
            
            return out
    
    def __call__(self, index):
        return next(self)
    
    def __len__(self):
        return self.length//self.stride + ((self.length % self.stride) > 0)
    
    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None


def BGR2RGBs(imgs):
    return np.stack([cv2.cvtColor(x.copy(), cv2.COLOR_BGR2RGB) for x in imgs], 0)


class Resampling:
    
    def __init__(self, out_fps=25, list_data=['boxes', ], key_indexs='image_inds'):
        self.list_data = list_data
        self.int_fps = float(out_fps)
        self.out_fps = float(out_fps)
        self.key_indexs = key_indexs
        assert self.key_indexs not in self.list_data
    
    def reset(self, int_fps):
        self.int_fps = float(int_fps)
        return self
    
    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids-0.5)*self.out_fps/self.int_fps + 1)), 0),
                     int(np.floor((ids+0.5)*self.out_fps/self.int_fps) + 1))
        return list(iout)
    
    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        out[self.key_indexs] = list()
        
        for index, ids in enumerate(inp[self.key_indexs]):
            ids_outs = self.compute_ids(ids)
            for i in ids_outs:
                out[self.key_indexs].append(i)
                for key in self.list_data:
                    out[key].append(inp[key][index])

        return out


class ReadingResampledVideo:

    def __init__(self, filename, out_fps, stride=1):
        self.filename = filename
        self.video_cap = None
        self.stride = stride
        self.out_fps = float(out_fps)
        self.in_fps = 0
        self.out_length = 0
        self.in_length = 0
        self.in_count = 0
        assert self.stride > 0

    def get_number_frames(self):
        return self.out_length

    def get_shape(self):
        return (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def get_fps(self):
        return self.out_fps

    def __enter__(self):
        self.video_cap = cv2.VideoCapture(self.filename)
        self.in_count = 0
        self.in_length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.in_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.out_length = int(self.in_length * self.out_fps / self.in_fps)
        return self

    def __iter__(self):
        return self

    def compute_ids(self, ids):
        iout = range(max(int(np.floor((ids - 0.5) * self.out_fps / self.in_fps + 1)), 0),
                     int(np.floor((ids + 0.5) * self.out_fps / self.in_fps) + 1))
        return list(iout)

    def __next__(self):
        out = {'frames_bgr': list(), 'frames_inds': list()}

        while len(out['frames_inds']) < self.stride:
            ret, img = self.video_cap.read()
            if not ret:
                break
            ids_outs = self.compute_ids(self.in_count)
            for i in ids_outs:
                out['frames_inds'].append(i)
                out['frames_bgr'].append(img)
            self.in_count = self.in_count + 1

        if len(out['frames_inds']) == 0:
            raise StopIteration

        return out

    def __call__(self, index):
        return next(self)

    def __len__(self):
        return self.out_length // self.stride + ((self.out_length % self.stride) > 0)

    def __exit__(self, type, value, tb):
        try:
            self.video_cap.release()
        except:
            pass
        self.video_cap = None


class MockFileBoxes:
    
    def __init__(self, fileboxes, list_data=['boxes', 'image_inds']):
        if isinstance(fileboxes, list):
            dat = [np.load(_, allow_pickle=True) for _ in fileboxes]
            self.image_inds = np.int64(dat[0]['image_inds'])
            for x, d in zip(fileboxes, dat):
                if not np.array_equal(self.image_inds, np.int64(d['image_inds'])):
                    print('error', x, self.image_inds.shape, d['image_inds'].shape)
                    #os.remove(x)
                    assert False
            self.data = dict()
            for key in list_data:
                for d in dat:
                    if key in d:
                        self.data[key] = d[key]
                        break
                assert key in self.data
            del dat
        else:
            dat = np.load(fileboxes, allow_pickle=True)
            self.data = {key: dat[key] for key in list_data}
            self.image_inds = dat['image_inds']
            del dat

    def reset(self):
        return self

    def __len__(self):
        return max(self.image_inds)+1
    
    def __enter__(self):
        return self
    
    def __call__(self, inp):
        if isinstance(inp, dict): 
            if 'frames_inds' in inp:
                ids = inp['frames_inds']
            else:
                ids = [inp['image_ind'], ]
                del inp['image_ind']
        elif isinstance(inp, list):
            ids = inp
            inp = dict()
        else:
            ids = [inp, ]
            inp = dict()
        
        val = [x in ids for x in self.image_inds]
        for key in self.data:
            inp[key] = self.data[key][val]
        
        return inp
    
    def __iter__(self):
        for count in range(len(self)):
            yield self(count)

    def __exit__(self, type, value, tb):
        pass


class FilterTrack:

    def __init__(self, list_data, filt_key, filt_list):
        self.filt_key = filt_key
        self.filt_list = filt_list
        self.list_data = list_data

    def reset(self):
        return self

    def __call__(self, inp):
        ii = [i for i in range(len(inp[self.filt_key])) if inp[self.filt_key][i] in self.filt_list]
        for k in self.list_data:
            inp[k] = [inp[k][i] for i in ii]

        return inp

class FilterValues:

    def __init__(self, condition, list_data=['boxes', 'image_inds', ], key_values='image_inds', list_pass=list()):
        self.list_data = list_data
        self.list_pass = list_pass
        self.condition = condition
        self.key_values = key_values

    def reset(self):
        return self

    def __call__(self, inp):
        out = {key: list() for key in self.list_data}
        for key in self.list_pass:
            out[key] = inp[key]

        for index, ids in enumerate(inp[self.key_values]):
            if self.condition(ids):
                for key in self.list_data:
                    out[key].append(inp[key][index])



        return out