import cv2
import numpy as np
from utils.config import VIDEO_RESIZE_DIM, NORMALIZATION_MEAN, NORMALIZATION_STD, MAX_FRAMES, FPS  # Import MAX_FRAMES and FPS

def resize_frame(frame, target_size=VIDEO_RESIZE_DIM):
    """Resizes a single frame to the target dimensions."""
    return cv2.resize(frame, target_size)

def normalize_frame(frame, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD):
    """Normalizes pixel values of a single frame."""
    normalized_frame = (frame.astype(np.float32) / 255.0 - mean) / std
    return normalized_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB
        resized_frame = resize_frame(frame)
        normalized_frame = normalize_frame(resized_frame)
        # Remove the uint8 conversion to maintain float values
        frames.append(normalized_frame)
    cap.release()
    return frames

def sample_frames(frames, max_frames=MAX_FRAMES, fps=FPS):
    """Samples a fixed number of frames from the video.

    Args:
        frames (list or np.ndarray): List or array of video frames.
        max_frames (int): Maximum number of frames to sample.
        fps (int): Target frames per second.

    Returns:
        np.ndarray: Sampled frames.
    """
    if not frames:
        return np.array([])

    original_fps = 30  # Assuming original video FPS is 30
    sampling_rate = original_fps / fps
    indices = np.linspace(0, len(frames) - 1, int(min(max_frames, len(frames) * sampling_rate)), dtype=int)
    sampled_frames = np.array(frames)[indices.astype(int)]
    return sampled_frames

if __name__ == '__main__':
    # Example usage
    sample_video_path = 'dataset/train/BSL_NZSL/word1/video1.mp4'  # Replace with a valid path
    processed_frames = process_video(sample_video_path)
    if processed_frames:
        print(f"Processed {len(processed_frames)} frames, shape: {processed_frames[0].shape}, dtype: {processed_frames[0].dtype}")
        sampled_frames = sample_frames(processed_frames)
        print(f"Sampled frames shape: {sampled_frames.shape}, dtype: {sampled_frames.dtype}")
    else:
        print("Video processing failed.")