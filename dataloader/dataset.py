import os
import torch
from torch.utils.data import Dataset
from .video_preprocessing import process_video, sample_frames
from .pose_estimation import PoseEstimator
from utils.config import MAX_FRAMES
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = sorted(os.listdir(os.path.join(root_dir, list(os.listdir(root_dir))[0])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = self._load_video_paths()
        self.pose_estimator = PoseEstimator()

    def _load_video_paths(self):
        video_paths = []
        for language_dir in os.listdir(self.root_dir):
            language_path = os.path.join(self.root_dir, language_dir)
            if os.path.isdir(language_path):
                for word_dir in os.listdir(language_path):
                    word_path = os.path.join(language_path, word_dir)
                    if os.path.isdir(word_path):
                        for video_file in os.listdir(word_path):
                            if video_file.endswith('.mp4'):
                                video_paths.append((os.path.join(word_path, video_file), self.class_to_idx[word_dir]))
        return video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        processed_frames = process_video(video_path)
        if processed_frames is None or len(processed_frames) == 0:
            raise ValueError(f"Could not load or process frames from {video_path}")

        sampled_frames = sample_frames(processed_frames)

        keypoint_sequences = []
        for frame in sampled_frames:
            keypoints = self.pose_estimator.extract_keypoints(frame)
            if keypoints is not None:
                keypoint_sequences.append(keypoints)
            else:
                keypoint_sequences.append(np.zeros(33*3 + 21*3 + 21*3 + 468*3)) # Pad with zeros

        # Pad or truncate sequences to a fixed length
        if len(keypoint_sequences) < self.sequence_length:
            padding = [np.zeros(33*3 + 21*3 + 21*3 + 468*3)] * (self.sequence_length - len(keypoint_sequences))
            keypoint_sequences.extend(padding)
        elif len(keypoint_sequences) > self.sequence_length:
            keypoint_sequences = keypoint_sequences[:self.sequence_length]

        keypoint_sequences = np.array(keypoint_sequences)

        if self.transform:
            keypoint_sequences = self.transform(keypoint_sequences)

        return torch.tensor(keypoint_sequences, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    # Example usage
    from utils.config import MAX_FRAMES
    train_dataset = SignLanguageDataset(root_dir='dataset/train', sequence_length=10)
    print(f"Number of training samples: {len(train_dataset)}")
    sample = train_dataset[0]
    print(f"Sample keypoint sequence shape: {sample[0].shape}")
    print(f"Sample label: {sample[1]}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    test_dataset = SignLanguageDataset(root_dir='dataset/test', sequence_length=10)
    print(f"Number of testing samples: {len(test_dataset)}")