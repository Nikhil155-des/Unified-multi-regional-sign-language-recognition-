import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import MobileNetV3Large
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Config
CONFIG = {
    'input_shape': (224, 224, 3),
    'num_frames': 30,  # We'll sample 30 frames from each video
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs': 50,
    'num_classes': 100,  # Assuming 100 words in the vocabulary
    'visual_feature_dim': 512,
    'motion_feature_dim': 256,
    'shared_feature_dim': 512,
    'languages': ['BSL_NZSL', 'ISL_Auslan'],
    'dataset_path': 'dataset/'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
