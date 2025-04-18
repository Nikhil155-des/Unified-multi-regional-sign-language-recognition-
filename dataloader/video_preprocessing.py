import cv2
import numpy as np
from utils.config import VIDEO_RESIZE_DIM, NORMALIZATION_MEAN, NORMALIZATION_STD

def resize_frame(frame, target_size=VIDEO_RESIZE_DIM):
    """Resizes a single frame to the target dimensions."""
    return cv2.resize(frame, target_size)

def normalize_frame(frame, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD):
    """Normalizes pixel values of a single frame."""
    normalized_frame = (frame.astype(np.float32) / 255.0 - mean) / std
    return normalized_frame

def process_video(video_path):
    """
    Reads a video, resizes and normalizes each frame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of processed frames (numpy arrays).
    """

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = resize_frame(frame)
        normalized_frame = normalize_frame(resized_frame)
        frames.append(normalized_frame)
    cap.release()
    return frames

if __name__ == '__main__':
    # Example usage
    sample_video_path = 'dataset/train/BSL_NZSL/word1/video1.mp4'  # Replace with a valid path
    processed_frames = process_video(sample_video_path)
    if processed_frames:
        print(f"Processed {len(processed_frames)} frames, shape: {processed_frames[0].shape}")
    else:
        print("Video processing failed.")