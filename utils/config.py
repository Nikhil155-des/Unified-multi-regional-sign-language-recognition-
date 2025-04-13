"""
Configuration settings for the Unified Multi-Regional Sign Language Recognition model.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATASET_DIR = ROOT_DIR / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train' / 'BSL_NZSL'
TEST_DIR = DATASET_DIR / 'test' / 'ISL_Auslan'
MODEL_SAVE_DIR = ROOT_DIR / 'saved_models'
LOG_DIR = ROOT_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [DATASET_DIR, TRAIN_DIR, TEST_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Video preprocessing
VIDEO_RESIZE_DIM = (224, 224)  # Width, Height
MAX_FRAMES = 150  # Maximum number of frames to consider from each video
FPS = 30  # Target FPS for video sampling
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZATION_STD = [0.229, 0.224, 0.225]  # ImageNet std

# MediaPipe settings
POSE_LANDMARKS = 33  # Number of pose landmarks
HAND_LANDMARKS = 21  # Number of landmarks per hand
FACE_LANDMARKS = 468  # Total face landmarks
SELECTED_FACE_LANDMARKS = 50  # Number of selected face landmarks to use

# Model parameters
VISUAL_FEATURE_DIM = 512
MOTION_FEATURE_DIM = 256
TEMPORAL_FEATURE_DIM = 256
SHARED_FEATURE_DIM = 512
NUM_CLASSES = 100  # Assuming 100 sign words in the dataset

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_INTERVAL = 5  # Save every N epochs

# Loss weights
CLASSIFIER_LOSS_WEIGHT = 1.0
DOMAIN_CONFUSION_LOSS_WEIGHT = 0.1

# Languages
LANGUAGES = {
    'train': ['BSL', 'NZSL'],
    'test': ['ISL', 'Auslan']
}

# Device settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed for reproducibility
RANDOM_SEED = 42