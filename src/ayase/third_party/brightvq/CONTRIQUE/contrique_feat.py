import torch
from .modules.network import get_network
from .modules.CONTRIQUE_model import CONTRIQUE_model
from .fragmentation import *
from torchvision import transforms
import numpy as np
import cv2
import scipy.ndimage
import os
import argparse
import pickle
from joblib import Parallel, delayed

from PIL import Image


opt = {
    'weight': 0.620,
    'phase': 'test',
    'anno_file': 'updated_Final_CSV_DATA_300_vimeo_sampled.csv',
    'data_prefix': './sampled_videos/',
    'sample_types': {
        'technical': {
            'fragments_h': 7,
            'fragments_w': 7,
            'fsize_h': 32,
            'fsize_w': 32,
            'aligned': 40,
            'clip_len': 40,
            't_frag': 20,
            'frame_interval': 2,
            'num_clips': 1
        }
    }
}



class CONTRIQUEFeatureExtractor():
    def __init__(self, model_path='models/CONTRIQUE_checkpoint25.tar', device=None, opt=opt):
        """
        Initialize the CONTRIQUE feature extractor.

        Parameters:
        - model_path (str): Path to the pre-trained CONTRIQUE model.
        - device (torch.device): Device to run the model on. Defaults to CUDA if available.
        """
        self.opt = opt
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        env_model_path = os.environ.get("AYASE_BRIGHTVQ_CONTRIQUE_MODEL_PATH")
        if env_model_path:
            resolved_model_path = env_model_path
        elif os.path.isabs(model_path):
            resolved_model_path = model_path
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_model_path = os.path.join(script_dir, model_path)

        # Check if the model file exists
        if not os.path.isfile(resolved_model_path):
            raise FileNotFoundError(f"Model file not found at {resolved_model_path}")

        self.model = self._load_model(resolved_model_path)
        self.transform = transforms.ToTensor()
        
        
        
    def _load_model(self, model_path):
        """
        Load the CONTRIQUE model from the specified path.

        Parameters:
        - model_path (str): Path to the pre-trained CONTRIQUE model.

        Returns:
        - model (torch.nn.Module): Loaded CONTRIQUE model.
        """
        encoder = get_network('resnet50', pretrained=False)
        model = CONTRIQUE_model(encoder, 2048)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    
    def spatio_temporal_fragmentation(self, frames):
        # get the spatio-temporal fragmentation
        data = ViewDecompositionDataset(self.opt, frames)
        try: 
            print(data)
            frames = data['technical']
            print(frames.shape)
        except Exception as e:
            print('Error in spatio_temporal_fragmentation', e)
            return None
        return frames
    
    def Y_compute_lnl(self, Y):
        if(len(Y.shape)==2):
            Y = np.expand_dims(Y,axis=2)

        maxY = scipy.ndimage.maximum_filter(Y,size=(17,17,1))
        minY = scipy.ndimage.minimum_filter(Y,size=(17,17,1))
        Y_scaled = -1+(Y-minY)* 2/(1e-3+maxY-minY)
        Y_transform =  np.exp(np.abs(Y_scaled)*4)-1
        Y_transform[Y_scaled<0] = -Y_transform[Y_scaled<0]
        
        # since values are fed in deep network, we need to normalize them to -1 to 1
        Y_transform = (Y_transform - np.min(Y_transform))/(np.max(Y_transform)-np.min(Y_transform))
        Y_transform = 2*Y_transform - 1
        
        return Y_transform

    
    def extract_features(self, frames, NLN=False, STF=False):
        """
        Extract features from a single frame or a batch of frames.

        :param frames: A single frame (H x W x C) or a batch of frames (N x H x W x C).
        :return: Extracted features as a NumPy array.
        """
        frames = np.array(frames)
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=0)  # Convert single frame to batch
        else:
            raise ValueError("Input frames should be a list og NumPy array")
        
        # Downscale images by 2
        frames_2 = [cv2.resize(frame, dsize = None, fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC) for frame in frames]
        
        if STF:
            # compute spatio-temporal fragmentation
            frames = self.spatio_temporal_fragmentation(frames)
            # tensor to numpy 
            frames = np.asarray([frame.numpy() for frame in frames])
            print(frames.shape)
            frames_2 = self.spatio_temporal_fragmentation(frames_2)
            frames_2 = np.asarray([frame.numpy() for frame in frames_2])
        
        if NLN:
            # compute nln in parallel
            frames = np.asarray(Parallel(n_jobs=1)(delayed(self.Y_compute_lnl)(frame) for frame in frames))
            frames_2 = np.asarray(Parallel(n_jobs=1)(delayed(self.Y_compute_lnl)(frame) for frame in frames_2))
        
        feats = np.zeros([frames.shape[0],4096])
        with torch.no_grad():
            for i in range(frames.shape[0]):

                image = torch.from_numpy(frames[i]).permute(2,0,1).unsqueeze(0).to(self.device)
                image_2 = torch.from_numpy(frames_2[i]).permute(2,0,1).unsqueeze(0).to(self.device)

                _,_, _, _, model_feat, model_feat_2, _, _ = self.model(image, image_2)
                feat = np.hstack((model_feat.detach().cpu().numpy(),\
                                        model_feat_2.detach().cpu().numpy()))
                # append to the list
                feats[i] = feat

        return feats
        
