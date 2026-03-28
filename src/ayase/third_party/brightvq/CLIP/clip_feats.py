import torch
from .clip import clip
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image


class CLIPFeatureExtractor():
    def __init__(self, device=None):
        """
        Initialize the CLIP Image feature extractor.

        Parameters:
        - device (torch.device): Device to run the model on. Defaults to CUDA if available.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = self._load_model()
        
    def _load_model(self):
        """
        Load the CLIP model.

        Parameters:

        Returns:
        - model (torch.nn.Module): Loaded CLIP model.
        """
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        from .clipiqa_arch import CLIPIQA 
        model = CLIPIQA()
        return model, preprocess
    
    def extract_features(self, frames, batch_size=80):
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
        
        # print number of frames 
        #print("Number of frames: ", frames.shape[0])    
        
        feats = []
        with torch.no_grad():
            # batch wise 
            for i in range(0, frames.shape[0], batch_size):
                # Get the features from the CLIP model
                images = frames[i:i+batch_size]
                images = [
                    self.preprocess(
                        Image.fromarray(
                            np.clip(image * 255.0, 0, 255).astype(np.uint8),
                            "RGB",
                        )
                    )
                    for image in images
                ]
                images = torch.stack(images).to(self.device)
                #feat = self.model.encode_image(images)
                feat = self.model(images)
                # extend the feats list 
                feats.extend(feat.cpu().numpy())
            # normalize the features
            feats = np.array(feats)
            feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
        return feats
        
