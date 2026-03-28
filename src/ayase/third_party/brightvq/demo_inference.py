"""
    BrightRate: HDR-UGC Video Quality Assessment Model
"""

import os
import imageio_ffmpeg as ffmpeg
import numpy as np 
import pandas as pd
import subprocess
import json
from tqdm import tqdm
import argparse
import torch
import time
from joblib import Parallel, delayed
import math
from sklearn.preprocessing import StandardScaler
import warnings 

warnings.filterwarnings('ignore')

from util_hdr_10bit import read_yuv_video, read_mp4_10bit

try:
    from mpi4py import MPI
except ImportError:
    class _SingleProcessComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            return None

    class _SingleProcessMPI:
        COMM_WORLD = _SingleProcessComm()

        @staticmethod
        def Finalize():
            return None

    MPI = _SingleProcessMPI()

class FeatureExtractor:
    """Base class for all feature extractors"""
    
    def __init__(self, name):
        self.name = name
        
    def extract_features(self, input_data):
        """Extract features from input data"""
        raise NotImplementedError("Subclasses must implement extract_features")

class ModelFactory:
    """Factory class to create feature extraction models"""
    
    @staticmethod
    def get_model(method):
        """Get the model for the specified method"""
        if method == 'contrique':    
            from CONTRIQUE.contrique_feat import CONTRIQUEFeatureExtractor
            feature_extractor = CONTRIQUEFeatureExtractor()
            return feature_extractor.extract_features
        
        elif method == 'hdr':
            from HDR.hdr_feat import HDR_feat
            return HDR_feat
        
        elif method == 'clip':
            from CLIP.clip_feats import CLIPFeatureExtractor
            feature_extractor = CLIPFeatureExtractor()
            return feature_extractor.extract_features
            
        else:
            raise ValueError(f'Method "{method}" not found.')

class VideoProcessor:
    """Class for processing videos and extracting frames"""
    
    def __init__(self, args):
        self.args = args
        
    def read_video_frames(self, video_path, width=None, height=None):
        """Read frames from a video file"""
        try:
            if self.args.read_yuv:
                if width is None or height is None:
                    raise ValueError("Width and height must be provided for YUV videos")
                
                frames = read_yuv_video(
                    video_path, width, height, rgb=True, 
                    n_jobs=self.args.num_workers, 
                    num_frames=self.args.num_frames, 
                    unique_frames=self.args.unique_frames
                )
            else: 
                frames = read_mp4_10bit(
                    video_path, 
                    ffmpeg_path=self.args.ffmpeg_path, 
                    rgb=True, 
                    n_jobs=self.args.num_workers,
                    num_frames=self.args.num_frames, 
                    unique_frames=self.args.unique_frames
                )
            
            return frames
        except Exception as e:
            print(f"Error reading video: {e}")
            return None
    
    def process_frame(self, frame, model):
        """Process a single frame with the given model"""
        try:
            return model(frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

class FeatureCombiner:
    """Class for combining features from different models"""
    
    @staticmethod
    def combine_features(contrique_feats, hdr_feats, clip_feats, expected_dim=None):
        """Combine features from different models into a single feature vector
        
        Args:
            contrique_feats: Features from CONTRIQUE model
            hdr_feats: Features from HDR model
            clip_feats: Features from CLIP model
            expected_dim: Expected feature dimension for the model
            
        Returns:
            Combined feature vector ready for quality prediction
        """
        try:
            # Process CONTRIQUE features
            contrique_feats = np.nan_to_num(contrique_feats.astype(np.float32))
            
            # Calculate temporal difference
            feat_diff = np.diff(contrique_feats, axis=0)
            # Pad with zeros at the beginning
            feat_diff = np.concatenate((np.zeros((1, *contrique_feats.shape[1:])), feat_diff), axis=0)
            # Concatenate original features with differences
            contrique_feats = np.concatenate((contrique_feats, feat_diff), axis=-1)
            
            # Average across frames if multi-dimensional
            if len(contrique_feats.shape) >= 2:
                contrique_feats = np.mean(contrique_feats, axis=0).flatten()
            
            # Normalize
            contrique_feats = contrique_feats.reshape(-1, 1)[:, 0]
            contrique_feats = StandardScaler().fit_transform(contrique_feats.reshape(-1, 1)).flatten()
            
            # Process HDR features
            hdr_feats = np.nan_to_num(hdr_feats.astype(np.float32))
            hdr_feats = StandardScaler().fit_transform(hdr_feats.reshape(-1, 1)).flatten()
            
            # Process CLIP features
            clip_feats = np.nan_to_num(clip_feats.astype(np.float32))
            clip_feats = StandardScaler().fit_transform(clip_feats.reshape(-1, 1)).flatten()
            if len(clip_feats.shape) >= 2:
                clip_feats = np.mean(clip_feats, axis=0).flatten()
            
            # Combine all features
            combined = np.concatenate((contrique_feats, hdr_feats, clip_feats), axis=0)
            
            # Adjust dimensions if needed
            if expected_dim is not None and combined.size != expected_dim:
                print(f"Warning: Feature dimension mismatch. Got {combined.size}, expected {expected_dim}")
                
                if combined.size > expected_dim:
                    # Truncate extra features
                    combined = combined[:expected_dim]
                    print(f"Truncated features to {expected_dim} dimensions")
                else:
                    # Pad with zeros
                    padding = np.zeros(expected_dim - combined.size)
                    combined = np.concatenate((combined, padding))
                    print(f"Padded features to {expected_dim} dimensions")
            
            return combined.reshape(1, -1)
            
        except Exception as e:
            print(f"Error combining features: {e}")
            return None

class QualityPredictor:
    """Class for predicting video quality scores"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()
        self.expected_features = self._get_expected_features()
        
    def _load_model(self):
        """Load the quality prediction model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Warning: Model file not found at {self.model_path}")
                return None
                
            model = torch.load(self.model_path, weights_only=False)
            print(f"Successfully loaded quality prediction model from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    def _get_expected_features(self):
        """Get the expected number of features for the model"""
        if self.model is None:
            return None
            
        # Try to extract expected feature count
        if hasattr(self.model, 'n_features_in_'):
            expected = self.model.n_features_in_
            print(f"Model expects {expected} features as input")
            return expected
        else:
            print("Could not determine expected feature count from model")
            return None
    
    def predict_quality(self, features):
        """Predict quality score from features"""
        if self.model is None:
            print("No quality prediction model available")
            return None
        
        try:
            # Check if features dimensions match what model expects
            if self.expected_features is not None:
                if features.shape[1] != self.expected_features:
                    print(f"Feature dimension mismatch: got {features.shape[1]}, expected {self.expected_features}")
                    
                    # Adjust dimensions
                    if features.shape[1] > self.expected_features:
                        features = features[:, :self.expected_features]
                    else:
                        # Pad with zeros
                        padding = np.zeros((features.shape[0], self.expected_features - features.shape[1]))
                        features = np.hstack((features, padding))
            
            predictions = self.model.predict(features)
            return predictions
        except Exception as e:
            print(f"Error predicting quality: {e}")
            return None

class HDRVideoQualityAssessor:
    """Main class for HDR video quality assessment"""
    
    def __init__(self, args):
        """Initialize the HDR Video Quality Assessor"""
        self.args = args
        self.video_processor = VideoProcessor(args)
        self.dataset = None
        self.models = {}
        
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        
        # Create save directory
        os.makedirs(os.path.join(self.args.save_path, self.args.method_qa), exist_ok=True)
        
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Load quality prediction model
        regressor_path = getattr(self.args, 'model_path', 
                       './models/brightrate_brighvq.pt')
        self.quality_predictor = QualityPredictor(regressor_path)
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        try:
            self.dataset = pd.read_csv(self.args.dataset_csv)
            print(f"Loaded dataset with {len(self.dataset)} videos")
            
            # Distribute dataset based on MPI or partitioning
            self._distribute_dataset()
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def _distribute_dataset(self):
        """Distribute dataset across MPI ranks or data parts"""
        if self.size > 1:
            if self.rank == 0:
                print(f"Running on {self.size} MPI ranks")
            self.dataset = self.dataset.iloc[self.rank::self.size]
            print(f"Rank {self.rank}: processing {len(self.dataset)} videos")
        else:
            print("Running on a single node")
            if self.args.data_part[0] != -1 and self.args.data_part[1] != -1:
                node_name = subprocess.check_output('hostname').decode('utf-8').strip()
                print(f"Running on {node_name} with part {self.args.data_part[0]+1} of {self.args.data_part[1]}")
                
                # Calculate slice indices
                start_idx = math.floor(len(self.dataset) * self.args.data_part[0] / self.args.data_part[1])
                end_idx = math.floor(len(self.dataset) * (self.args.data_part[0] + 1) / self.args.data_part[1])
                
                if start_idx >= len(self.dataset):
                    print(f"Start index {start_idx} out of bounds for dataset length {len(self.dataset)}.")
                    return
                
                if end_idx > len(self.dataset):
                    end_idx = len(self.dataset)
                    
                self.dataset = self.dataset.iloc[start_idx:end_idx]
                self.dataset.reset_index(drop=True, inplace=True)
                print(f"Processing {len(self.dataset)} videos (part {self.args.data_part[0]+1}/{self.args.data_part[1]})")
    
    def load_models(self):
        """Load all feature extraction models"""
        try:
            self.models["contrique"] = ModelFactory.get_model("contrique")
            self.models["clip"] = ModelFactory.get_model("clip")
            self.models["hdr"] = ModelFactory.get_model("hdr")
            print("Successfully loaded all feature extraction models")
            return True
        except Exception as e:
            print(f"Error loading feature extraction models: {e}")
            return False
    
    def process_videos(self):
        """Process all videos in the dataset"""
        if self.dataset is None or len(self.dataset) == 0:
            print("No videos to process. Dataset is empty.")
            return
            
        if not self.models:
            print("No feature extraction models loaded.")
            return
        
        for i in tqdm(range(len(self.dataset)), desc=f"Processing videos (Rank {self.rank})"):
            self.process_video(i)
        
        # Ensure all MPI processes finish
        try:
            self.comm.Barrier()
        except Exception as e:
            print(f"MPI Barrier error: {e}")
        
        print(f"Rank {self.rank}: Feature extraction completed")
    
    def process_video(self, idx):
        """Process a single video, extract features and predict quality"""
        # Get video information
        video_name = self.dataset["Video"][idx]
        ext = '.yuv' if self.args.read_yuv else '.mp4'
        video_path = os.path.join(self.args.video_path, video_name + ext)
        save_file = os.path.join(self.args.save_path, self.args.method_qa, video_name + '.npy')
        score_file = os.path.join(self.args.save_path, self.args.method_qa, video_name + '_score.txt')
        
        # Skip if already processed and not overwriting
        if os.path.exists(save_file) and os.path.exists(score_file) and not getattr(self.args, 'overwrite', False):
            print(f'Skipping video {video_name} (features already computed)')
            #return
        
        # Get video dimensions if needed
        width = self.dataset['width'][idx] if 'width' in self.dataset.columns else None
        height = self.dataset['height'][idx] if 'height' in self.dataset.columns else None
        
        # Show processing message
        display_name = self.dataset["name"][idx] if "name" in self.dataset.columns else video_name
        print(f'Computing video {display_name}: {video_path}')
        
        # Read video frames
        frames = self.video_processor.read_video_frames(video_path, width, height)
        if frames is None:
            print(f"Could not read frames for video {video_name}")
            return
        
        # Extract features from each model
        features = {}
        for model_name, model in self.models.items():
            try:
                if self.args.frame_wise and self.args.parallel_level == 'frame':
                    # Process frames in parallel
                    model_features = Parallel(n_jobs=self.args.num_workers)(
                        delayed(self.video_processor.process_frame)(frame, model) for frame in frames
                    )
                else:
                    # Process all frames together
                    model_features = model(frames)
                
                if model_features is None or len(model_features) == 0:
                    print(f'No features computed for video {display_name} with model {model_name}')
                    return
                
                features[model_name] = np.asarray(model_features)
                
                # Save individual model features
                if getattr(self.args, 'save_individual', False):
                    model_save_dir = os.path.join(self.args.save_path, f"{self.args.method_qa}_{model_name}")
                    os.makedirs(model_save_dir, exist_ok=True)
                    model_save_file = os.path.join(model_save_dir, video_name + '.npy')
                    np.save(model_save_file, model_features)
                    
            except Exception as e:
                print(f"Error extracting {model_name} features for video {video_name}: {e}")
                return
        
        # Combine features
        expected_dim = self.quality_predictor.expected_features if self.quality_predictor else None
        combined_features = FeatureCombiner.combine_features(
            features["contrique"], features["hdr"], features["clip"], expected_dim
        )
        
        if combined_features is None:
            print(f"Error combining features for video {video_name}")
            return
            
        # Predict quality score
        predictions = self.quality_predictor.predict_quality(combined_features)
        if predictions is not None:
            print(f"Predicted score for video {video_name}: {predictions}")
            
            # Save prediction to file
            try:
                with open(score_file, 'w') as f:
                    f.write(f"{predictions[0]:.6f}")
            except Exception as e:
                print(f"Error saving prediction: {e}")
        
        # Save combined features
        try:
            np.save(save_file, combined_features)
        except Exception as e:
            print(f"Error saving features: {e}")
    
    def inspect_features(self, idx=0):
        """
        Inspect feature dimensions for a sample video
        Useful for debugging dimension mismatches
        """
        if not self.load_models():
            return
            
        if self.dataset is None and not self.load_dataset():
            return
            
        # Get video information
        video_name = self.dataset["Video"][idx]
        ext = '.yuv' if self.args.read_yuv else '.mp4'
        video_path = os.path.join(self.args.video_path, video_name+ext)
        
        # Get video dimensions if needed
        width = self.dataset['width'][idx] if 'width' in self.dataset.columns else None
        height = self.dataset['height'][idx] if 'height' in self.dataset.columns else None
        
        print(f"\nInspecting features for video: {video_name}")
        print("="*60)
        
        # Read a small sample of frames
        old_frames = self.args.num_frames
        self.args.num_frames = min(10, old_frames if old_frames > 0 else 100)
        
        frames = self.video_processor.read_video_frames(video_path, width, height)
        if frames is None:
            print("Could not read frames for inspection")
            self.args.num_frames = old_frames
            return
            
        print(f"Read {len(frames)} frames for inspection")
        
        # Extract and report feature dimensions
        features = {}
        for model_name, model in self.models.items():
            try:
                if self.args.frame_wise and self.args.parallel_level == 'frame':
                    model_features = Parallel(n_jobs=1)(
                        delayed(self.video_processor.process_frame)(frame, model) for frame in frames[:2]
                    )
                else:
                    model_features = model(frames)
                    
                features[model_name] = np.asarray(model_features)
                
                print(f"{model_name} features shape: {features[model_name].shape}")
                print(f"{model_name} features size: {features[model_name].size}")
                
            except Exception as e:
                print(f"Error extracting {model_name} features: {e}")
        
        # Restore original settings
        self.args.num_frames = old_frames
        
        # Report expected model dimensions
        if self.quality_predictor.expected_features:
            print(f"\nQuality prediction model expects {self.quality_predictor.expected_features} features")
            
        print("\nFeature inspection complete")
    
    def run(self):
        """Run the complete quality assessment pipeline"""
        if not self.load_dataset():
            return False
            
        if not self.load_models():
            return False
            
        # Run feature inspection if requested
        if getattr(self.args, 'inspect', False):
            self.inspect_features()
            return True
            
        self.process_videos()
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HDR Video Quality Assessment Framework"
    )
    parser.add_argument('--dataset_csv', type=str, 
                    default='./sample_videos.csv', 
                    help='Path to the csv file containing video information')
    parser.add_argument('--read_yuv', action='store_true', 
                    help='If True, read YUV videos')
    parser.add_argument('--video_path', type=str, 
                    default='./sample_videos/', 
                    help='Path to the videos')
    parser.add_argument('--save_path', type=str, 
                    default='./output_feats/', 
                    help='Path to save the features')
    parser.add_argument('--method_qa', type=str, 
                    default='brightrate', 
                    help='Name of the method')
    parser.add_argument('--num_frames', type=int, 
                    default=-1, 
                    help='Number of frames to sample. -1 means all frames')
    parser.add_argument('--unique_frames', type=bool, 
                    default=False, 
                    help='If True, sample unique frames')
    parser.add_argument('--frame_wise', type=bool, 
                    default=False, 
                    help='If True, extract frame-wise features')
    parser.add_argument('--gpu', type=int, 
                    default=0, 
                    help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, 
                    default=1, 
                    help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, 
                    default=-1, 
                    help='Number of workers for parallel processing')
    parser.add_argument('--ffmpeg_path', type=str, 
                    default='./ffmpeg-ffprobe/', 
                    help='Path to ffmpeg executable')
    parser.add_argument('--parallel_level', type=str, 
                    default='frame', 
                    choices=['frame', 'video'], 
                    help='Parallelize at video or frame level')
    parser.add_argument('--curr_data_part', type=int, 
                    default=-1, 
                    help='Current data part index (0-based)')
    parser.add_argument('--total_data_part', type=int, 
                    default=-1, 
                    help='Total number of data parts')
    parser.add_argument('--model_path', type=str,
                    default='./models/brightrate_brightvq.pt',
                    help='Path to the quality prediction model')
    parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing feature files')
    parser.add_argument('--save_individual', action='store_true',
                    help='Save features from individual models')
    parser.add_argument('--inspect', action='store_true',
                    help='Run feature inspection mode')
    
    args = parser.parse_args()
    args.data_part = [args.curr_data_part, args.total_data_part]
    return args

def print_banner(args):
    """Print formatted configuration information"""
    print('\n' + '='*80)
    print('HDR Video Quality Assessment Framework')
    print('='*80)
    print(f'Dataset: {args.dataset_csv}')
    print(f'Video format: {"YUV" if args.read_yuv else "MP4"}')
    print(f'Video path: {args.video_path}')
    print(f'Method: {args.method_qa}')
    print(f'Output path: {args.save_path}')
    print(f'GPU: {args.gpu}')
    print(f'Parallel level: {args.parallel_level}')
    print(f'Workers: {args.num_workers}')
    if args.data_part[0] != -1:
        print(f'Processing part {args.data_part[0]+1} of {args.data_part[1]}')
    if args.inspect:
        print('Mode: Feature inspection')
    if args.overwrite:
        print('Overwriting existing files')
    print('='*80 + '\n')

def main():
    # Parse command line arguments
    args = parse_arguments()
    print_banner(args)
    
    try:
        # Create and run the quality assessor
        assessor = HDRVideoQualityAssessor(args)
        success = assessor.run()
        
        if success:
            print("Feature extraction and quality assessment completed successfully.")
        else:
            print("Feature extraction and quality assessment failed.")
            
    except Exception as e:
        print(f"An error occurred during quality assessment: {e}")
    finally:
        # Always finalize MPI
        try:
            MPI.Finalize()
        except:
            pass  # Already finalized or not initialized

if __name__ == '__main__':
    main()

