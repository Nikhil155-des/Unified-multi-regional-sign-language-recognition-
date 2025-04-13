"""
Video preprocessing utilities for sign language recognition.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Union
import sys
import os
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

def load_video(video_path: str, max_frames: int = MAX_FRAMES, target_fps: int = FPS) -> np.ndarray:
    """
    Load video file and sample frames to achieve target FPS.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        target_fps: Target frames per second
        
    Returns:
        Numpy array of frames with shape (num_frames, height, width, channels)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling rate to achieve target FPS
    if original_fps > target_fps:
        sample_rate = original_fps / target_fps
    else:
        sample_rate = 1  # Use all frames if original FPS is lower
    
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames and frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames based on calculated rate
        if frame_idx % sample_rate < 1:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
        
        # Stop if we've collected enough frames
        if len(frames) >= max_frames:
            break
    
    cap.release()
    
    # Pad with last frame if needed
    while len(frames) < max_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Ensure we only return max_frames
    frames = frames[:max_frames]
    
    return np.array(frames)

def resize_frames(frames: np.ndarray, target_size: Tuple[int, int] = VIDEO_RESIZE_DIM) -> np.ndarray:
    """
    Resize video frames to target dimensions.
    
    Args:
        frames: Video frames with shape (num_frames, height, width, channels)
        target_size: Target frame size (width, height)
        
    Returns:
        Resized frames with shape (num_frames, target_height, target_width, channels)
    """
    num_frames, height, width, channels = frames.shape
    resized_frames = np.zeros((num_frames, target_size[1], target_size[0], channels), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        resized_frames[i] = cv2.resize(frame, target_size)
    
    return resized_frames

def normalize_frames(frames: np.ndarray, mean: List[float] = NORMALIZATION_MEAN, 
                    std: List[float] = NORMALIZATION_STD) -> np.ndarray:
    """
    Normalize pixel values using ImageNet mean and std.
    
    Args:
        frames: Video frames with shape (num_frames, height, width, channels)
        mean: Mean values for RGB channels
        std: Standard deviation values for RGB channels
        
    Returns:
        Normalized frames with shape (num_frames, channels, height, width)
    """
    # Convert to float
    frames = frames.astype(np.float32) / 255.0
    
    # Transpose from (frames, H, W, C) to (frames, C, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))
    
    # Apply normalization
    for i in range(3):  # RGB channels
        frames[:, i] = (frames[:, i] - mean[i]) / std[i]
    
    return frames

def preprocess_video(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete video preprocessing pipeline.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (normalized_frames, original_frames)
        - normalized_frames: Tensor of shape (num_frames, channels, height, width)
        - original_frames: Numpy array of shape (num_frames, height, width, channels)
    """
    # Load video
    frames = load_video(video_path)
    
    # Keep original frames for keypoint extraction
    original_frames = frames.copy()
    
    # Resize frames
    frames = resize_frames(frames)
    
    # Normalize frames
    normalized_frames = normalize_frames(frames)
    
    # Convert to tensor
    normalized_frames = torch.from_numpy(normalized_frames)
    
    return normalized_frames, original_frames