"""Configuration settings for the Unified Multi-Regional Sign Language Recognition model."""

import os
from pathlib import Path
import torch

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATASET_DIR = "D:/Engeneering/Third Year/SEM6/Unified-multi-regional-sign-language-recognition-/dataset"
TRAIN_DIR = "D:/Engeneering/Third Year/SEM6/Unified-multi-regional-sign-language-recognition-/dataset/train/BSL_NZSL"
TEST_DIR = "D:/Engeneering/Third Year/SEM6/Unified-multi-regional-sign-language-recognition-/dataset/test/ISL_Auslan"
MODEL_SAVE_DIR = "D:/Engeneering/Third Year/SEM6/Unified-multi-regional-sign-language-recognition-/saved_models"
LOG_DIR = "D:/Engeneering/Third Year/SEM6/Unified-multi-regional-sign-language-recognition-/logs"

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
TOTAL_KEYPOINT_DIM = POSE_LANDMARKS * 3 + HAND_LANDMARKS * 3 * 2 + SELECTED_FACE_LANDMARKS * 3

# Model parameters
VISUAL_FEATURE_DIM = 512  # Feature dimension from ResNet/MobileNetV3 [cite: 12]
MOTION_FEATURE_DIM = 256  # Feature dimension from TCN [cite: 15]
TEMPORAL_FEATURE_DIM = 256 # GRU hidden dimension
MLSLT_FEATURE_DIM = 512 # MLSLT output dimension [cite: 23]
MLSLT_HIDDEN_DIM = 512
MLSLT_NUM_LAYERS = 2
MLSLT_DROPOUT = 0.1
NUM_CLASSES = 56 # Assuming 100 sign words in the dataset
TCN_OUTPUT_SEQUENCE_LENGTH = 10 # Output frames from TCN

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 1
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