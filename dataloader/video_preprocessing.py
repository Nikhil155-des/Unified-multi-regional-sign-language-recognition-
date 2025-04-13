import cv2
import mediapipe as mp

def preprocess_video(video_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame / 255.0  # normalization
        frames.append(frame)
    cap.release()
    return frames
