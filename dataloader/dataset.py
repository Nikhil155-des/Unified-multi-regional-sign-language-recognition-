"""
PyTorch dataset classes for multi-regional sign language recognition.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Union
import sys

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *
from data_loader.video_preprocessing import preprocess_video
from data_loader.pose_estimator import extract_keypoints_from_video, normalize_keypoints

class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language videos.
    """
    def __init__(self, root_dir: str, languages: List[str] = None, 
                 transform=None, is_train: bool = True, max_frames: int = MAX_FRAMES,
                 precompute_features: bool = False):
        """
        Args:
            root_dir: Root directory containing sign language videos
            languages: List of languages to include 
            transform: Optional transform to apply to video frames
            is_train: Whether this is a training dataset
            max_frames: Maximum number of frames to include per video
            precompute_features: Whether to precompute and cache features
        """
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.transform = transform
        self.max_frames = max_frames
        self.precompute_features = precompute_features
        
        # Set languages based on train/test mode if not specified
        if languages is None:
            self.languages = LANGUAGES['train'] if is_train else LANGUAGES['test']
        else:
            self.languages = languages
            
        self.samples = []
        self.cache = {}  # For cached features
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset from directory."""
        word_classes = set()
        
        # Support for multiple file extensions
        valid_extensions = ['.mp4', '.avi', '.mov']
        
        # Scan directory for video files
        for file_path in self.root_dir.glob('**/*.*'):
            if file_path.suffix.lower() not in valid_extensions:
                continue
                
            # Extract word from filename (assuming format: word.mp4 or word_variant.mp4)
            word = file_path.stem.split('_')[0]
            word_classes.add(word)
            
            # Get language from file path
            lang = None
            for language in self.languages:
                if language.lower() in str(file_path).lower():
                    lang = language
                    break
            
            if lang is None:
                # Try to infer from directory structure
                for parent_dir in file_path.parents:
                    for language in self.languages:
                        if language.lower() in parent_dir.name.lower():
                            lang = language
                            break
                    if lang:
                        break
                        
            if lang is None:
                continue  # Skip if language not identified
                
            self.samples.append({
                'path': str(file_path),
                'word': word,
                'language': lang
            })
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(word_classes))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.num_classes = len(self.word_to_idx)
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        print(f"Found {self.num_classes} word classes")
        
        # If requested, pre-compute features (this can take time but speeds up training)
        if self.precompute_features:
            print("Pre-computing features (this may take a while)...")
            for i, sample in enumerate(self.samples):
                if i % 10 == 0:
                    print(f"Processing sample {i+1}/{len(self.samples)}")
                video_path = sample['path']
                self._process_and_cache_video(video_path)
    
    def _process_and_cache_video(self, video_path):
        """Process video and cache the results."""
        if video_path in self.cache:
            return
        
        try:
            # Load and preprocess video
            normalized_frames, original_frames = preprocess_video(video_path)
            
            # Extract keypoints
            keypoints = extract_keypoints_from_video(original_frames)
            keypoints = normalize_keypoints(keypoints)
            
            # Store in cache
            self.cache[video_path] = {
                'frames': normalized_frames,
                'keypoints': torch.tensor(keypoints, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Create empty entries to avoid repeated processing attempts
            empty_frames = torch.zeros((MAX_FRAMES, 3, VIDEO_RESIZE_DIM[1], VIDEO_RESIZE_DIM[0]), 
                                     dtype=torch.float32)
            empty_keypoints = torch.zeros((MAX_FRAMES, get_full_keypoint_dim()), 
                                         dtype=torch.float32)
            self.cache[video_path] = {
                'frames': empty_frames,
                'keypoints': empty_keypoints
            }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        word = sample['word']
        language = sample['language']
        
        # Check if we have cached features
        if self.precompute_features and video_path in self.cache:
            frames_tensor = self.cache[video_path]['frames']
            keypoints_tensor = self.cache[video_path]['keypoints']
        else:
            try:
                # Load and preprocess video
                normalized_frames, original_frames = preprocess_video(video_path)
                
                # Extract keypoints
                keypoints = extract_keypoints_from_video(original_frames)
                keypoints = normalize_keypoints(keypoints)
                
                # Convert to tensors
                frames_tensor = normalized_frames
                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
                
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                # Return dummy tensors on error
                frames_tensor = torch.zeros((MAX_FRAMES, 3, VIDEO_RESIZE_DIM[1], VIDEO_RESIZE_DIM[0]), 
                                          dtype=torch.float32)
                keypoints_tensor = torch.zeros((MAX_FRAMES, get_full_keypoint_dim()), 
                                             dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        
        # Convert label to tensor
        label = torch.tensor(self.word_to_idx.get(word, 0), dtype=torch.long)
        
        # One-hot encode language for domain confusion
        lang_idx = self.languages.index(language) if language in self.languages else 0
        lang_onehot = torch.zeros(len(self.languages))
        lang_onehot[lang_idx] = 1.0
        
        return {
            'frames': frames_tensor,
            'keypoints': keypoints_tensor,
            'label': label,
            'language': lang_onehot,
            'word': word,
            'lang_name': language
        }

def get_data_loaders(batch_size=BATCH_SIZE, precompute_features=False):
    """
    Create train and test data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        precompute_features: Whether to precompute features
        
    Returns:
        Tuple of (train_loader, test_loader, num_classes)
    """
    # Create datasets
    train_dataset = SignLanguageDataset(
        root_dir=TRAIN_DIR,
        languages=LANGUAGES['train'],
        is_train=True,
        precompute_features=precompute_features
    )
    
    test_dataset = SignLanguageDataset(
        root_dir=TEST_DIR,
        languages=LANGUAGES['test'],
        is_train=False,
        precompute_features=precompute_features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.num_classes