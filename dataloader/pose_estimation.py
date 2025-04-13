"""
Pose estimation utilities for sign language recognition using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Union
import sys
import os
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe models with optimal parameters for sign language
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_landmarks(frame: np.ndarray) -> np.ndarray:
    """
    Extract pose landmarks from a single frame.
    
    Args:
        frame: RGB frame with shape (height, width, channels)
        
    Returns:
        Array of pose landmarks with shape (POSE_LANDMARKS * 3,)
    """
    results = pose.process(frame)
    landmarks = []
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        # If no pose detected, pad with zeros
        landmarks = [0.0] * (POSE_LANDMARKS * 3)
    
    return np.array(landmarks)

def extract_hand_landmarks(frame: np.ndarray) -> np.ndarray:
    """
    Extract hand landmarks from a single frame for both hands.
    
    Args:
        frame: RGB frame with shape (height, width, channels)
        
    Returns:
        Array of hand landmarks with shape (HAND_LANDMARKS * 3 * 2,)
    """
    results = hands.process(frame)
    landmarks = []
    
    if results.multi_hand_landmarks:
        # Process each detected hand (up to 2 hands)
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx < 2:  # Only use up to 2 hands
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # If only one hand detected, pad with zeros for the second hand
        if len(results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0] * (HAND_LANDMARKS * 3))
    else:
        # If no hands detected, pad with zeros for both hands
        landmarks = [0.0] * (HAND_LANDMARKS * 3 * 2)
    
    return np.array(landmarks)

def extract_face_landmarks(frame: np.ndarray) -> np.ndarray:
    """
    Extract selected face landmarks from a single frame.
    
    Args:
        frame: RGB frame with shape (height, width, channels)
        
    Returns:
        Array of face landmarks with shape (SELECTED_FACE_LANDMARKS * 3,)
    """
    results = face_mesh.process(frame)
    landmarks = []
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Define indices for important facial features
        important_indices = [
            # Eyes (edges and pupils)
            *range(33, 42),      # Left eye
            *range(263, 272),    # Right eye
            # Eyebrows
            *range(70, 80),      # Left eyebrow
            *range(300, 310),    # Right eyebrow
            # Lips and mouth
            *range(61, 69),      # Outer lips
            *range(291, 299),    # Inner lips
            # Nose
            *range(168, 175),    # Nose tip and bridge
        ]
        
        # Select only the first SELECTED_FACE_LANDMARKS indices
        important_indices = important_indices[:SELECTED_FACE_LANDMARKS]
        
        for idx in important_indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        # If no face detected, pad with zeros
        landmarks = [0.0] * (SELECTED_FACE_LANDMARKS * 3)
    
    return np.array(landmarks)

def extract_keypoints_from_frame(frame: np.ndarray) -> np.ndarray:
    """
    Extract all keypoints (pose, hands, face) from a single frame.
    
    Args:
        frame: RGB frame with shape (height, width, channels)
        
    Returns:
        Array of all keypoints concatenated
    """
    pose_landmarks = extract_pose_landmarks(frame)
    hand_landmarks = extract_hand_landmarks(frame)
    face_landmarks = extract_face_landmarks(frame)
    
    # Concatenate all landmarks
    all_landmarks = np.concatenate([pose_landmarks, hand_landmarks, face_landmarks])
    
    return all_landmarks

def extract_keypoints_from_video(frames: np.ndarray) -> np.ndarray:
    """
    Extract keypoints from all frames in a video.
    
    Args:
        frames: Video frames with shape (num_frames, height, width, channels)
        
    Returns:
        Array of keypoints with shape (num_frames, num_keypoints)
    """
    num_frames = frames.shape[0]
    all_keypoints = []
    
    for i in range(num_frames):
        frame_keypoints = extract_keypoints_from_frame(frames[i])
        all_keypoints.append(frame_keypoints)
    
    return np.array(all_keypoints)

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to be invariant to position and scale.
    
    Args:
        keypoints: Array of keypoints with shape (num_frames, num_keypoints)
        
    Returns:
        Normalized keypoints with the same shape
    """
    normalized = np.copy(keypoints)
    
    # Process each frame separately
    for i in range(keypoints.shape[0]):
        frame_keypoints = keypoints[i].reshape(-1, 3)  # Reshape to (N, 3) for x, y, z
        
        # Skip frames with all zeros
        if np.all(frame_keypoints == 0):
            continue
        
        # Find non-zero keypoints
        non_zero_mask = np.any(frame_keypoints != 0, axis=1)
        non_zero_keypoints = frame_keypoints[non_zero_mask]
        
        if len(non_zero_keypoints) == 0:
            continue
        
        # Calculate center as mean of non-zero keypoints
        center = np.mean(non_zero_keypoints, axis=0)
        
        # Calculate scale as max distance from center
        distances = np.linalg.norm(non_zero_keypoints - center, axis=1)
        scale = np.max(distances) if np.max(distances) > 0 else 1.0
        
        # Normalize by subtracting center and dividing by scale
        normalized_frame = (frame_keypoints - center) / scale
        
        # Update the normalized array
        normalized[i] = normalized_frame.flatten()
    
    return normalized

def get_full_keypoint_dim():
    """
    Calculate the total dimension of keypoints.
    
    Returns:
        Total number of values in the keypoint vector
    """
    pose_dim = POSE_LANDMARKS * 3
    hands_dim = HAND_LANDMARKS * 3 * 2  # Two hands
    face_dim = SELECTED_FACE_LANDMARKS * 3
    
    return pose_dim + hands_dim + face_dim