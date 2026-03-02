import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class BackgroundDiversityModule(PipelineModule):
    name = "background_diversity"
    description = "Checks background complexity (entropy) to detect concept bleeding"
    default_config = {
        "min_entropy_threshold": 3.0,
        "use_rembg": True,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.min_entropy_threshold = self.config.get("min_entropy_threshold", 3.0)
        self.use_rembg = self.config.get("use_rembg", True)
        self._rembg_session = None
        self._available = False

    def setup(self):
        if self.use_rembg:
            try:
                from rembg import new_session
                # Initialize session (downloads u2net model if needed)
                self._rembg_session = new_session()
                self._available = True
                logger.info("rembg loaded for background analysis.")
            except ImportError:
                logger.warning("rembg not installed. Background diversity check disabled.")
            except Exception as e:
                logger.warning(f"rembg init failed: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            from rembg import remove
            
            # rembg expects RGB numpy or PIL
            # Remove background. Returns RGBA where A=0 is background.
            # Wait, rembg removes background (makes it transparent).
            # We want the *background* content. 
            # Actually, rembg is a matting tool. It gives us the Mask.
            # We can use the mask to Isolate the background.
            
            # Run rembg
            # result = remove(image, session=self._rembg_session) # Returns RGBA of Foreground
            
            # We need the mask. 
            # rembg has 'remove' but maybe we can just get the mask?
            # Standard usage: remove(input_data) -> output_data (FG with alpha)
            # The alpha channel IS the mask (255=FG, 0=BG).
            
            # Convert image to RGB for rembg
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image

            output_rgba = remove(img_rgb, session=self._rembg_session)
            
            # Extract Mask (Alpha)
            alpha = output_rgba[:, :, 3]
            
            # Invert Mask to get Background (0=FG, 255=BG)
            bg_mask = 255 - alpha
            
            # Analyze Entropy of the Background regions
            # We only care about pixels where mask > 0
            
            # Convert original image to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Compute histogram of background pixels
            # Safety: if entire image is FG, bg_mask is all 0.
            if np.sum(bg_mask) < 100:
                # No background?
                return sample
                
            # entropy calculation
            entropy = self._calculate_shannon_entropy(hsv, bg_mask)
            
            # Store/Validate
            # High entropy = Complex/Diverse background (Good for disentanglement?)
            # Low entropy = Plain, uniform background (Studio shot).
            # Evaluating...md says: "High entropy is required to disentangle the subject from the environment."
            
            if entropy < self.min_entropy_threshold:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Background Diversity (Entropy: {entropy:.2f})",
                        details={"bg_entropy": float(entropy)},
                        recommendation="Dataset may have 'Concept Bleeding' if subject is always on simple backgrounds."
                    )
                )

            # --- M-PSNR (Background Stability) ---
            # Compare this frame's background with previous/next if available?
            # Since process(sample) usually looks at one sample, we might need to load multiple frames.
            # Let's load 2 frames if it is a video.
            
            if sample.is_video:
                frames = self._load_frames(sample, num_frames=2)
                if len(frames) == 2:
                    f1, f2 = frames[0], frames[1]
                    
                    # Get masks for both
                    # Note: Need to run rembg on both. This is heavy.
                    # Optimization: Only do this if strictly enabled.
                    
                    # Assume f1, f2 are RGB
                    mask1 = 255 - remove(f1, session=self._rembg_session)[:,:,3]
                    mask2 = 255 - remove(f2, session=self._rembg_session)[:,:,3]
                    
                    # Intersect masks: consider pixels that are BG in BOTH frames
                    common_bg_mask = cv2.bitwise_and(mask1, mask2)
                    
                    # Calculate MSE on common BG
                    if np.sum(common_bg_mask) > 100:
                        diff = cv2.absdiff(f1, f2)
                        # Mask diff
                        diff_bg = cv2.bitwise_and(diff, diff, mask=common_bg_mask)
                        
                        mse = np.sum(diff_bg.astype(np.float32) ** 2) / (np.sum(common_bg_mask) * 3 + 1e-7)
                        psnr = 10 * np.log10(255**2 / (mse + 1e-7))
                        
                        # Threshold? High PSNR = Stable BG. Low PSNR = Moving BG.
                        if psnr < 30.0: # Arbitrary threshold, typical video compression is ~30-40
                             sample.validation_issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.INFO,
                                    message=f"Unstable Background (M-PSNR: {psnr:.2f})",
                                    details={"m_psnr": float(psnr)},
                                    recommendation="Background is moving or flickering. Subject disentanglement might be poor."
                                )
                            )

        except Exception as e:
            logger.warning(f"Background analysis failed: {e}")
            
        return sample

    def _load_frames(self, sample: Sample, num_frames: int = 2) -> list:
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames for background diversity: {e}")
        return frames

    def _calculate_shannon_entropy(self, img, mask):
        # Calculate histogram for each channel restricted by mask
        entropy = 0
        for i in range(3):
            # 256 bins for 0-255
            hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
            # Normalize to probability
            prob = hist / (np.sum(hist) + 1e-7)
            # Filter zero entries
            prob = prob[prob > 0]
            # H = -sum(p * log2(p))
            entropy -= np.sum(prob * np.log2(prob))
            
        return entropy / 3.0 # Average entropy per channel

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None
