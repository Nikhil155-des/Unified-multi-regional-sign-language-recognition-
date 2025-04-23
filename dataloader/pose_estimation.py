import cv2
import numpy as np
import mediapipe as mp
from utils.config import POSE_LANDMARKS, HAND_LANDMARKS, FACE_LANDMARKS, SELECTED_FACE_LANDMARKS

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)  # Add root to sys.path

from utils.config import VIDEO_RESIZE_DIM, NORMALIZATION_MEAN, NORMALIZATION_STD, MAX_FRAMES, FPS
class PoseEstimator:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, frame):
        """Extracts pose, hand, and face keypoints from a single video frame.

        Args:
            frame (np.ndarray): A single video frame (RGB image).

        Returns:
            np.ndarray: An array of shape (1404,) containing the flattened
                        x, y, and visibility coordinates of pose (33*3),
                        left hand (21*3), and right hand (21*3), and face (468*3)
                        keypoints, or None if no pose is detected. We are selecting only 50 face landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        keypoints = []

        # Pose landmarks
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (POSE_LANDMARKS * 3))

        # Left hand landmarks
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (HAND_LANDMARKS * 3))

        # Right hand landmarks
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (HAND_LANDMARKS * 3))

        # Face landmarks (select only 50)
        if results.face_landmarks:
            for i in range(SELECTED_FACE_LANDMARKS): # Use config value
                landmark = results.face_landmarks.landmark[i]
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (SELECTED_FACE_LANDMARKS * 3))

        if keypoints:
            return np.array(keypoints).flatten()
        else:
            return None

    def __del__(self):
        self.holistic.close()

if __name__ == '__main__':
    # Example usage
    from utils.config import VIDEO_RESIZE_DIM
    estimator = PoseEstimator()
    sample_frame = np.zeros((VIDEO_RESIZE_DIM[1], VIDEO_RESIZE_DIM[0], 3), dtype=np.uint8)  # Replace with a real frame
    keypoints = estimator.extract_keypoints(sample_frame)
    if keypoints is not None:
        print(f"Extracted keypoints shape: {keypoints.shape}")  # Should be (33*3 + 21*3 + 21*3 + 50*3)
        print(f"First few keypoints: {keypoints[:10]}")
    else:
        print("No pose detected in the frame.")