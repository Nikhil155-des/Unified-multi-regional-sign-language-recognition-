import torch
from torch.utils.data import Dataset
import numpy as np
from .video_preprocessing import process_video, sample_frames
from utils.config import MAX_FRAMES, TOTAL_KEYPOINT_DIM, TRAIN_DIR, TEST_DIR  # Import TRAIN_DIR, TEST_DIR
from .pose_estimation import PoseEstimator
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)  # Add root to sys.path

from utils.config import VIDEO_RESIZE_DIM, NORMALIZATION_MEAN, NORMALIZATION_STD, MAX_FRAMES, FPS

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, sequence_length=30, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.classes = self._get_classes()  # Get classes (word directories)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_data = self._load_video_data()  # Load video paths and labels

    def _get_classes(self):
        """Gets the list of class names (word directories)."""
        classes = sorted(os.listdir(self.root_dir))
        print(f"Classes found: {classes}")  # Debugging
        return classes

    def _load_video_data(self):
        """Loads video paths and corresponding labels."""
        video_data = []
        for word_dir in os.listdir(self.root_dir):
            word_path = os.path.join(self.root_dir, word_dir)
            if os.path.isdir(word_path):
                for video_file in os.listdir(word_path):
                    if video_file.endswith(".mp4"):
                        video_path = os.path.join(word_path, video_file)
                        label = self.class_to_idx[word_dir]
                        video_data.append((video_path, label))
                        print(f"Found: {video_path}, Label: {label}")  # Debugging
        print(f"Total videos loaded: {len(video_data)}")  # Debugging
        return video_data

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_path, label = self.video_data[idx]
        processed_frames = process_video(video_path)
        processed_frames = sample_frames(processed_frames, max_frames=MAX_FRAMES)

        keypoint_sequences = []
        pose_estimator = PoseEstimator()  # Initialize inside __getitem__
        for frame in processed_frames:
            keypoints = pose_estimator.extract_keypoints(frame)
            if keypoints is not None:
                keypoint_sequences.append(keypoints)
            else:
                keypoint_sequences.append(np.zeros(TOTAL_KEYPOINT_DIM))

        # Pad or truncate to sequence length
        if len(keypoint_sequences) < self.sequence_length:
            padding = [np.zeros(TOTAL_KEYPOINT_DIM)] * (self.sequence_length - len(keypoint_sequences))
            keypoint_sequences.extend(padding)
        elif len(keypoint_sequences) > self.sequence_length:
            keypoint_sequences = keypoint_sequences[:self.sequence_length]

        keypoint_sequences = np.array(keypoint_sequences, dtype=np.float32)  # Ensure float32

        if self.transform:
            keypoint_sequences = self.transform(keypoint_sequences)

        return torch.tensor(keypoint_sequences, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Example Usage
    train_dataset = SignLanguageDataset(root_dir=TRAIN_DIR, sequence_length=30)
    print("Train Dataset:")
    print(f"  Number of samples: {len(train_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Class to index: {train_dataset.class_to_idx}")

    if len(train_dataset) > 0:
        sample_keypoints, sample_label = train_dataset[0]
        print(f"  Sample keypoints shape: {sample_keypoints.shape}")
        print(f"  Sample label: {sample_label}")

    test_dataset = SignLanguageDataset(root_dir=TEST_DIR, sequence_length=30)
    print("\nTest Dataset:")
    print(f"  Number of samples: {len(test_dataset)}")