""" 
Python Script to read HDR 10-bit video files and return a list of NumPy arrays representing each frame.
Since, some of the videos can be in HDR 10-bit format, we need to read them in a different way than the normal 8-bit videos.

NOTE: MP4 videos are normalized and whereas webm videos are not normalized. 

In case of 8-bit videos, we can use the following code to read the video:
    import imageio
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    print(f"Read {len(frames)} frames from the video.")

Or alternatively, we can use opecv to read the video:
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print(f"Read {len(frames)} frames from the video.")

"""
import os
import imageio_ffmpeg as ffmpeg
import numpy as np 
import subprocess
import json
import re
import cv2
from joblib import Parallel, delayed


def _resolve_ffmpeg_exe(ffmpeg_path):
    if ffmpeg_path:
        if os.path.isdir(ffmpeg_path):
            candidate = os.path.join(ffmpeg_path, "ffmpeg.exe")
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(ffmpeg_path, "ffmpeg")
            if os.path.exists(candidate):
                return candidate
        return ffmpeg_path
    return ffmpeg.get_ffmpeg_exe()


def _probe_pix_fmt(video_path, ffmpeg_path):
    ffmpeg_exe = _resolve_ffmpeg_exe(ffmpeg_path)
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    text = result.stderr or ""
    match = re.search(r"Video:.*?,\s*([a-zA-Z0-9_]+),\s*\d+x\d+", text)
    if match:
        return match.group(1)
    return "yuv420p"

#-------------------------------------------------**********-------------------------------------------------# 
def check_video_range(video_path):
    """
    Check the range of an MP4 video file.

    Parameters:
    - video_path: str, path to the video file

    Returns:
    - str, either 'tv' or 'pc' based on the range used in the video file, or 'unknown' if the range could not be determined
    """
    try:
        # Run ffprobe to get video stream information in JSON format
        cmd = [
            ffmpeg_path+'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_streams', 
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Parse the JSON output from ffprobe
        ffprobe_output = json.loads(result.stdout)
        
        # Get the color range from the first video stream
        streams = ffprobe_output.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'video':
                color_range = stream.get('color_range')
                if color_range:
                    return color_range
                else:
                    return 'unknown'
        return 'unknown'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'unknown'

#-------------------------------------------------**********-------------------------------------------------#
def verify_frames(frames):
    """
    Verify if a list of NumPy arrays correctly represents an HDR 10-bit video.

    Parameters:
    - frames: list of NumPy arrays, each representing a frame in the video

    Returns:
    - bool, True if the frames correctly represent an HDR 10-bit video, False otherwise
    """
    if not frames:
        print("No frames to verify.")
        return False

    # Check the data type of the frames
    if frames[0].dtype != np.float32:
        print("Incorrect data type.")
        return False

    # Check the range of pixel values
    min_value = np.min([np.min(frame) for frame in frames])
    max_value = np.max([np.max(frame) for frame in frames])
    
    if min_value < 0 or max_value > 1:
        print(f"Incorrect pixel value range: min={min_value}, max={max_value}")
        return False
    
    # Check if there are any values above the 8-bit maximum
    max_8bit_value = 255 / 1023
    if max_value <= max_8bit_value:
        print(f"No values above 8-bit maximum: max value={max_value}")
        return False

    print("The frames correctly represent an HDR 10-bit video.")
    return True

#-------------------------------------------------**********-------------------------------------------------#


def yuv2rgb_bt2020(y,u,v):
    # cast to float32 for yuv2rgb in BT2020
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    cb = u - 512
    cr = v - 512

    r = y+1.4747*cr
    g = y-0.1645*cb-0.5719*cr
    b = y+1.8814*cb

    r = r.astype(np.uint16)
    g = g.astype(np.uint16)
    b = b.astype(np.uint16)

    frame = np.stack((r,g,b), axis=2)

    return frame

#-------------------------------------------------**********-------------------------------------------------#

#-------------------------------------------------**********-------------------------------------------------#

def read_frame(video_path, frame_idx, width, height, bytes_per_pixel, bit_depth, pix_fmt, offset, scale, rgb, ffmpeg_path):
    """
    Reads and processes a single specific frame from the video using FFmpeg.
    """
    ffmpeg_exe = _resolve_ffmpeg_exe(ffmpeg_path)
    cmd = [
        ffmpeg_exe,
        '-i', video_path,
        '-vf', f"select='eq(n\\,{frame_idx})'",
        '-vsync', 'vfr',
        '-f', 'image2pipe',
        '-pix_fmt', pix_fmt,
        '-vcodec', 'rawvideo', '-'
    ]
    
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Read raw frame data
    raw_frame = pipe.stdout.read(int(width * height * bytes_per_pixel))
    pipe.stdout.close()
    pipe.wait()

    if not raw_frame:
        return None

    # Convert raw frame data to a NumPy array
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    image = np.frombuffer(raw_frame, dtype=dtype)

    # Separate the Y, U, and V planes
    y_plane = image[:width * height].reshape((height, width))
    u_plane = image[width * height:width * height + (width // 2) * (height // 2)].reshape((height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)
    v_plane = image[width * height + (width // 2) * (height // 2):].reshape((height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)

    if rgb:
        # Convert YUV to RGB (function `yuv2rgb_bt2020` assumed to be defined elsewhere)
        image = yuv2rgb_bt2020(y_plane, u_plane, v_plane)
    else:
        # Stack Y, U, and V planes into a 3D array
        image = np.stack((y_plane, u_plane, v_plane), axis=2)

    # Normalize the pixel values
    image = image.astype(np.float32)
    image = (image - offset) * scale
    image = np.clip(image, 0, 1)

    return image

def read_mp4_10bit(video_path, num_frames, pix_range='tv', ffmpeg_path='', rgb=True, n_jobs=-1, unique_frames=False):
    """
    Reads a specified number of equally spaced frames from a 10-bit MP4 video using FFmpeg.
    
    Parameters:
    - video_path: Path to the input video file.
    - num_frames: Number of frames to sample from the entire video.
    - pix_range: 'tv' or 'full' for normalization range.
    - ffmpeg_path: Path to the ffmpeg executable.
    - rgb: Boolean to convert the frames to RGB or keep as YUV.
    
    Returns:
    - frames: List of processed video frames as NumPy arrays (either YUV or RGB).
    """
    cap = cv2.VideoCapture(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video dimensions for {video_path}")
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    pix_fmt = _probe_pix_fmt(video_path, ffmpeg_path)

    # Determine bytes per pixel and bit depth
    if pix_fmt in ['yuv420p', 'yuvj420p']:
        bytes_per_pixel = 1.5
        bit_depth = 8
    elif pix_fmt in ['yuv420p10le', 'yuv420p10be']:
        bytes_per_pixel = 3
        bit_depth = 10
    elif pix_fmt in ['rgb48le', 'rgb48be']:
        bytes_per_pixel = 6
        bit_depth = 16

    # Set normalization scaling factors
    if pix_range == 'tv':
        offset = 64
        scale = 1 / (940 - 64)
    else:  # full range
        offset = 0
        scale = 1 / (1023 - 0)

    # Calculate frame indices to sample
    if unique_frames:
        frame_list = list(set(num_frames))
    else:
        if num_frames > total_frames or num_frames == -1:
            num_frames = total_frames  # Limit to total frames available
        frame_list = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []

    # Read frames in parallel using joblib
    frames = Parallel(n_jobs=n_jobs)(
        delayed(read_frame)(
            video_path, frame_idx, width, height, bytes_per_pixel, bit_depth, pix_fmt, offset, scale, rgb, ffmpeg_path
        )
        for frame_idx in frame_list
    )

    # Remove any None values in case of partial reads
    frames = [frame for frame in frames if frame is not None]
    

    return frames



#-------------------------------------------------**********-------------------------------------------------#

#-------------------------------------------------**********-------------------------------------------------#
def fread(fid, nelements, dtype, frame=True):
    data_array = np.fromfile(fid, dtype, nelements)
    if frame:
        data_array.shape = (nelements, 1)   
    
    return data_array

#-------------------------------------------------**********-------------------------------------------------#

def read_yuv_video_old(filename, width, height, pix_range='tv', rgb=True):
    """
    Optimized method to read all frames from a 10-bit YUV HDR video file at once.
    
    Parameters:
    - filename: path to the YUV file.
    - width: width of the YUV frames.
    - height: height of the YUV frames.
    
    Returns:
    - frames: a list of NumPy arrays, each representing a frame in RGB or YUV format.
    """
    # Define the scaling factors based on the range type
    if pix_range == 'tv':
        offset = 64
        scale = 1 / (940 - 64)
    else:  # full range
        offset = 0
        scale = 1 / (1023 - 0)

    # Calculate frame size for 10-bit YUV 4:2:0 (assuming 16-bit storage for 10-bit data)
    y_plane_size = height * width
    uv_plane_size = (height // 2) * (width // 2)  # YUV 4:2:0 chroma subsampling
    frame_size = (y_plane_size + 2 * uv_plane_size)  # YUV frame (Y + U + V)

    # Read the entire YUV file at once
    file_size = os.path.getsize(filename)
    num_frames = file_size // (frame_size * 2)  # *2 because 10-bit values are stored in 16 bits

    # Open and read the entire file as a single binary block
    with open(filename, 'rb') as file_object:
        raw_data = np.fromfile(file_object, dtype=np.uint16, count=file_size // 2)  # uint16 because of 10-bit in 16-bit storage
    
    # Split the raw data into frames
    frames = []
    for frame_num in range(num_frames):
        # Extract Y, U, and V components from the raw data
        start = frame_num * frame_size
        end_y = start + y_plane_size
        end_u = end_y + uv_plane_size
        end_v = end_u + uv_plane_size

        y = np.reshape(raw_data[start:end_y], (height, width))
        u = np.reshape(raw_data[end_y:end_u], (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)
        v = np.reshape(raw_data[end_u:end_v], (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)

        if rgb:
            # Convert YUV to RGB for the current frame
            frame = yuv2rgb_bt2020(y, u, v)
        else:
            # Stack Y, U, and V into a single frame
            frame = np.stack((y, u, v), axis=2)

        # Normalize the pixel values for HDR range
        frame = frame.astype(np.float32)
        frame = (frame - offset) * scale
        frame = np.clip(frame, 0, 1)

        # Append the frame to the list
        frames.append(frame)

    return frames


#-------------------------------------------------**********-------------------------------------------------#

def fread_chunk(filename, frame_num, y_plane_size, uv_plane_size, width, height, frame_size, scale, offset, rgb=True, normalize=True):
    """
    Function to process a single frame or a chunk of raw YUV data.
    
    Parameters:
    - raw_data: The entire raw YUV data.
    - frame_num: The index of the frame being processed.
    - y_plane_size: The size of the Y plane in the frame.
    - uv_plane_size: The size of the U or V plane in the frame.
    - width: Width of the YUV frame.
    - height: Height of the YUV frame.
    - frame_size: Total size of one YUV frame.
    - scale: Scaling factor for pixel normalization.
    - offset: Offset for normalization.
    - rgb: Boolean to decide whether to convert to RGB or keep as YUV.

    Returns:
    - frame: A single processed frame (either in RGB or YUV format).
    """
    # Skip to the start of the frame number
    with open(filename, 'rb') as file_object:
        file_object.seek(frame_num * frame_size * 2)  # *2 because 10-bit data is stored in 16-bit format
        # read the actual data
        raw_data = np.fromfile(file_object, dtype=np.uint16, count= frame_size)  # uint16 because of 10-bit in 16-bit storage

    # Extract Y, U, and V components from the raw data
    start = 0
    end_y = start + y_plane_size
    end_u = end_y + uv_plane_size
    end_v = end_u + uv_plane_size
    
    y = np.reshape(raw_data[start:end_y], (height, width))
    u = np.reshape(raw_data[end_y:end_u], (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)
    v = np.reshape(raw_data[end_u:end_v], (height // 2, width // 2)).repeat(2, axis=0).repeat(2, axis=1)

    if rgb:
        # Convert YUV to RGB
        frame = yuv2rgb_bt2020(y, u, v)
    else:
        # Stack Y, U, and V into a single frame
        frame = np.stack((y, u, v), axis=2)

    # Normalize the pixel values for HDR range
    frame = frame.astype(np.float32)
    if normalize:
        frame = (frame - offset) * scale
        frame = np.clip(frame, 0, 1)
    
    return frame


def read_yuv_video(filename, width, height, pix_range='tv', rgb=True, n_jobs=-1, num_frames=-1, unique_frames=False, normalize=True):
    """
    Optimized method to read all frames from a 10-bit YUV HDR video file using parallel processing with joblib.
    
    Parameters:
    - filename: path to the YUV file.
    - width: width of the YUV frames.
    - height: height of the YUV frames.
    - n_jobs: Number of parallel jobs (-1 means use all available CPUs).
    - num_frames: Number of frames to read (default is -1 to read all frames).
    
    Returns:
    - frames: a list of NumPy arrays, each representing a frame in RGB or YUV format.
    """

    # Define the scaling factors based on the range type
    if pix_range == 'tv':
        offset = 64
        scale = 1 / (940 - 64)
    else:  # full range
        offset = 0
        scale = 1 / (1023 - 0)

    # Calculate frame size for 10-bit YUV 4:2:0 (assuming 16-bit storage for 10-bit data)
    y_plane_size = height * width
    uv_plane_size = (height // 2) * (width // 2)  # YUV 4:2:0 chroma subsampling
    frame_size = (y_plane_size + 2 * uv_plane_size)  # YUV frame (Y + U + V)

    # Size and frame num list
    file_size = os.path.getsize(filename)
    total_frames = file_size // (frame_size * 2)  # *2 because 10-bit values are stored in 16 bits
    if unique_frames:
        frame_list = list(set(num_frames))
    else:
        if num_frames > total_frames or num_frames == -1:
            num_frames = total_frames
        frame_list = np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()

    # Parallel processing of frames using joblib

    frames = Parallel(n_jobs=n_jobs)(
        delayed(fread_chunk)(filename, frame_num, y_plane_size, uv_plane_size, width, height, frame_size, scale, offset, rgb, normalize=normalize)
        for frame_num in frame_list
    )

    return frames

#-------------------------------------------------**********-------------------------------------------------#
