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
import os
import glob
import tqdm


class ComputeParallelDict:
    def __init__(self, keys_in, key_out, dict_op):
        self.keys_in = keys_in
        self.key_out = key_out
        self.dict_op = dict_op
        self.list_name = list(self.dict_op.keys())

    def reset(self, list_name=None, **key):
        if list_name is not None:
            self.list_name = [_ for _ in list_name if _ in self.dict_op]
        else:
            self.list_name = list(self.dict_op.keys())

        for name in self.list_name:
            self.dict_op[name].reset(**key)

        return self

    def save(self):
        for name in self.dict_op:
            self.dict_op[name].save()

    def __call__(self, inp):
        out = dict()
        for name in self.list_name[:-1]:
            inp_ = {_: inp[_] for _ in self.keys_in}
            out[name] = self.dict_op[name](inp_)[self.key_out]

        inp = self.dict_op[self.list_name[-1]](inp)
        out[self.list_name[-1]] = inp[self.key_out]
        del inp[self.key_out]
        for name in self.list_name:
            inp[self.key_out+'_'+name] = out[name]
            if name == '':
                inp[self.key_out] = out[name]

        return inp


class ComputeDistance:
    def __init__(self, key_feat, folder_refs, normalize=True):
        self.key_feat = 'embs_' + key_feat
        self.fileall = os.path.join(folder_refs, 'embs_tracks.npz')
        if os.path.isfile(self.fileall):
            self.list_refs = dict(np.load(self.fileall))
            for filename in self.list_refs:
                print(filename, key_feat, self.list_refs[filename].shape)
        else:
            self.list_refs = dict()
            for folder_vid in glob.glob(os.path.join(folder_refs, '*')):
                if os.path.isdir(folder_vid):
                    key = os.path.basename(folder_vid)
                    listfile = glob.glob(os.path.join(folder_vid, 'embs_track*.npz'))
                    if len(listfile) > 0:
                        self.list_refs[key] = np.concatenate([np.load(filename)[self.key_feat] for filename in listfile], 0)
                        print(key, key_feat, self.list_refs[key].shape)
        self.list_insize = list(self.list_refs.keys())
        self.mu = 0.0
        self.sigma = 1.0
        self.normalize = normalize
        if self.normalize:
            file_statistics = os.path.join(folder_refs, 'statistics.npz')
            if os.path.isfile(file_statistics):
                print('Loading statistics from ', file_statistics)
                print('If you want recompute the statistics, delete the file', file_statistics)
                self.mu = np.load(file_statistics)['mu']
                self.sigma = np.load(file_statistics)['sigma']
            else:
                print('Computing statistics on the reference videos .....', flush=True)
                cum_mean = 0.0
                cum_vqm = 0.0
                cum_dem = 0.0

                for filename in self.list_refs:
                    feat = self.list_refs[filename]
                    self.reset(filename)
                    cum_mean_vid = 0.0
                    cum_vqm_vid = 0.0
                    cum_dem_vid = 0.0
                    for split in tqdm.tqdm(range(0, len(feat), 100)):
                        dist = self.compute_dist(feat[split:(split+100)], normalize=False)
                        cum_mean_vid += np.sum(dist)
                        cum_vqm_vid += np.sum(dist * dist)
                        cum_dem_vid += len(dist)
                    cum_mean += cum_mean_vid / cum_dem_vid
                    cum_vqm += cum_vqm_vid / cum_dem_vid
                    cum_dem += 1

                self.mu = cum_mean / cum_dem
                self.sigma = np.sqrt(cum_vqm / cum_dem - (self.mu**2))
                try:
                    np.savez(file_statistics, mu=self.mu, sigma=self.sigma)
                    print('Saving statistics to ', file_statistics)
                except:
                    pass
            print('Statistics:', self.mu, self.sigma, flush=True)
        self.list_insize = list(self.list_refs.keys())

    def reset(self, filename=None):
        if filename is None:
            self.list_insize = list(self.list_refs.keys())
            return self

        exclude = os.path.splitext(filename.split('/')[-1])[0]
        self.list_insize = list()
        for key in self.list_refs:
            if exclude == key:
                print('exclude:', key, flush=True)
            else:
                self.list_insize.append(key)

        return self

    def compute_dist(self, feats, normalize):
        num_feat = feats.shape[0]
        out_dist = np.nan * np.ones(num_feat)

        if num_feat == 0:
            return out_dist

        for filename in self.list_insize:
            ref_embs = self.list_refs[filename][None, :, :]
            dist = np.min(np.sum(np.square(ref_embs - feats[:, None, :]), -1), -1)
            out_dist = np.fmin(out_dist, dist)

        if normalize:
            out_dist = (out_dist-self.mu)/self.sigma

        return out_dist

    def save(self):
        if not os.path.isfile(self.fileall):
            print(f'INFO: save {self.fileall}!')
            np.savez(self.fileall, **self.list_refs)
        else:
            print(f'WARNING: {self.fileall} already exits!')

    def __call__(self, inp):
        embs = inp[self.key_feat]
        del inp[self.key_feat]
        if len(embs) == 0:
            inp['embs_dists'] = list()
        else:
            inp['embs_dists'] = self.compute_dist(np.stack(embs, 0), normalize=self.normalize)

        return inp


class ComputeDistanceMulti:
    def __init__(self, dict_folder_lambda):
        self.key_feats = list(dict_folder_lambda.keys())
        self.dict_lambda = {key_feat: dict_folder_lambda[key_feat][1] for key_feat in self.key_feats}
        self.list_refs = dict()
        for key_feat in self.key_feats:
            folder_refs = dict_folder_lambda[key_feat][0]
            fileall = os.path.join(folder_refs, 'embs_tracks.npz')
            if os.path.isfile(fileall):
                loc_list_refs = dict(np.load(fileall))
                for key in loc_list_refs:
                    if key not in self.list_refs:
                        self.list_refs[key] = dict()
                    self.list_refs[key][key_feat] = loc_list_refs[key]
            else:
                for folder_vid in glob.glob(os.path.join(folder_refs, '*')):
                    if os.path.isdir(folder_vid):
                        key = os.path.basename(folder_vid)
                        listfile = glob.glob(os.path.join(folder_vid, 'embs_track*.npz'))
                        if len(listfile) > 0:
                            feat = np.concatenate([np.load(filename)[key_feat] for filename in listfile], 0)
                            if key not in self.list_refs:
                                self.list_refs[key] = dict()
                            self.list_refs[key][key_feat] = feat
                            print(key_feat, key, self.list_refs[key][key_feat].shape)
        self.list_insize = None

    def reset(self, filename=None):
        if filename is None:
            self.list_insize = list(self.list_refs.keys())
            return self

        exclude = os.path.splitext(filename.split('/')[-1])[0]
        self.list_insize = list()
        for key in self.list_refs:
            if exclude == key:
                print('exclude:', key)
            else:
                self.list_insize.append(key)

        return self

    def __call__(self, inp):
        num_feat = len(inp['embs_'+self.key_feats[0]])
        if num_feat == 0:
            for key_feat in self.key_feats:
                del inp['embs_' + key_feat]
            inp['embs_dists'] = list()
            return inp
        inp['embs_dists'] = np.zeros(num_feat)

        for key_feat in self.key_feats:
            embs = inp['embs_'+key_feat]
            del inp['embs_'+key_feat]
            embs = np.stack(embs, 0)[:, None, :]
            dist = np.nan * np.ones(num_feat)
            for key in self.list_insize:
                if key_feat in self.list_refs[key]:
                    ref_embs = self.list_refs[key][key_feat][None, :, :]
                    dist = np.fmin(dist, np.min(np.sum(np.square(ref_embs - embs), -1), -1))
            inp['embs_dists'] += self.dict_lambda[key_feat] * dist
        return inp


def elab_boxes(boxes):
    boxes = np.stack(boxes, 1)
    return [np.min(boxes[0]), np.min(boxes[1]), np.max(boxes[2]), np.max(boxes[3])]


def elab_points(points):
    return np.stack(points, 0)


class ComputeTemporal:

    def __init__(self, time, stride, dict_functions):
        self.time = time
        self.stride = stride
        self.dict_functions = dict_functions
        self.dict_functions['boxes'] = elab_boxes
        self.dict_functions['points'] = elab_points
        self.dict_feats = {key: dict() for key in self.dict_functions}
        self.dict_count = dict()
        self.dict_inds = dict()
        assert self.stride <= self.time

    def reset(self):
        self.dict_feats = {key: dict() for key in self.dict_functions}
        self.dict_count = dict()
        self.dict_inds = dict()
        return self

    def num_tracks(self):
        return len(self.dict_count)

    def __call__(self, inp):
        out = {'embs_track': list(), 'embs_range': list(), 'embs_boxes': list()}
        for key in self.dict_feats:
            out['embs_'+key] = list()

        for index in range(len(inp['id_track'])):
            i = inp['id_track'][index]
            t = inp['image_inds'][index]
            if i < 0:
                continue
            if i in self.dict_count:
                if self.dict_inds[i] == t:
                    for key in self.dict_feats:
                        self.dict_feats[key][i].append(inp[key][index])
                    self.dict_count[i] = self.dict_count[i]+1
                    self.dict_inds[i] = t+1
                else:
                    for key in self.dict_feats:
                        self.dict_feats[key][i] = [inp[key][index], ]
                    self.dict_count[i] = 1
                    self.dict_inds[i] = t+1
            else:
                for key in self.dict_feats:
                    self.dict_feats[key][i] = [inp[key][index], ]
                self.dict_count[i] = 1
                self.dict_inds[i] = t+1

        for i in self.dict_count:
            while self.dict_count[i] >= self.time:
                t0 = self.dict_inds[i] - self.dict_count[i]
                self.dict_count[i] = self.dict_count[i] - self.stride
                out['embs_range'].append([t0, t0 + self.time])
                out['embs_track'].append(i)

                for key in self.dict_feats:
                    f0 = self.dict_functions[key](self.dict_feats[key][i][:self.time])
                    out['embs_'+key].append(f0)
                    self.dict_feats[key][i] = self.dict_feats[key][i][self.stride:]

        return out

class ComputeMean:
    def __init__(self, time, norm=True):
        self.time = time
        self.norm = norm
        self.eps = 1e-10

    def __call__(self, feats):
        length = len(feats)
        if length > self.time:
            s = (length-self.time)//2
            feats = feats[s:s+self.time]
        feats = np.asarray(feats)
        if self.norm:
            feats = feats / np.sqrt(np.sum(feats ** 2, -1, keepdims=True) + self.eps)
        return np.mean(feats, 0)


class PassIdentity:

    def __init__(self):
        pass

    def reset(self):
        return self

    def __call__(self, inp):
        return inp


class ComputeNormAvg:

    def __init__(self, key_feat, time=50, norm=True):
        self.time = time
        self.stride = time
        self.eps = 1e-10
        self.norm = norm
        self.key_feat = key_feat
        self.dict_feats = dict()
        self.dict_boxes = dict()
        self.dict_inds = dict()

    def reset(self):
        self.dict_feats = dict()
        self.dict_boxes = dict()
        self.dict_inds = dict()
        return self

    def num_tracks(self):
        return len(self.dict_feats)

    def elab_feats(self, feats):
        feats = np.asarray(feats)
        if self.norm:
            feats = feats / np.sqrt(np.sum(feats ** 2, -1, keepdims=True) + self.eps)
        return np.mean(feats, 0)

    def elab_boxes(self, boxes):
        boxes = np.stack(boxes, 1)
        return [np.min(boxes[0]), np.min(boxes[1]), np.max(boxes[2]), np.max(boxes[3])]

    def __call__(self, inp):
        out = {'embs_track': list(), 'embs_feats': list(), 'embs_range': list(), 'embs_boxes': list()}

        for i, t, f, b in zip(inp['id_track'], inp['image_inds'], inp[self.key_feat], inp['boxes']):
            if i < 0:
                continue
            if i in self.dict_feats:
                if self.dict_inds[i] == t:
                    self.dict_feats[i].append(f)
                    self.dict_boxes[i].append(b)
                    self.dict_inds[i] = t+1
                else:
                    self.dict_feats[i] = [f, ]
                    self.dict_boxes[i] = [b, ]
                    self.dict_inds[i] = t+1
            else:
                self.dict_feats[i] = [f, ]
                self.dict_boxes[i] = [b, ]
                self.dict_inds[i] = t+1

        for i in self.dict_feats:
            while len(self.dict_feats[i]) >= self.time:
                t0 = self.dict_inds[i] - len(self.dict_feats[i])
                out['embs_feats'].append(self.elab_feats(self.dict_feats[i][:self.time]))
                out['embs_boxes'].append(self.elab_boxes(self.dict_boxes[i][:self.time]))
                out['embs_range'].append([t0, t0 + self.time])
                out['embs_track'].append(i)

                self.dict_feats[i] = self.dict_feats[i][self.stride:]
                self.dict_boxes[i] = self.dict_boxes[i][self.stride:]

        return out
