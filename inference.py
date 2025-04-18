import torch
import torch.nn as nn
from dataloader.video_preprocessing import process_video
from dataloader.pose_estimation import PoseEstimator
from model.temporal_cnn import TemporalCNN
from model.mlslt import MLSLT
from model.visual_extractor import VisualFeatureExtractor
from utils.config import * # Import all configs
import cv2
import numpy as np

def inference(video_path, model_tcn, model_visual, model_mlslt, classifier, device, sequence_length=30):
    """
    Performs inference on a given video.

    Args:
        video_path (str): Path to the input video.
        model_tcn (nn.Module): Trained TemporalCNN model.
        model_visual (nn.Module): Trained VisualFeatureExtractor model.
        model_mlslt (nn.Module): Trained MLSLT model.
        classifier (nn.Module): Trained SignClassifier (transfer head).
        device (torch.device): Device to use for inference.
        sequence_length (int): Expected sequence length.

    Returns:
        str: The predicted sign word.
    """

    model_tcn.eval()
    model_visual.eval()
    model_mlslt.eval()
    classifier.eval()

    processed_frames = process_video(video_path)
    if not processed_frames:
        raise ValueError(f"Could not process video from {video_path}")

    pose_estimator = PoseEstimator()
    keypoint_sequences = []
    for frame in processed_frames:
        keypoints = pose_estimator.extract_keypoints(frame)
        if keypoints is not None:
            keypoint_sequences.append(keypoints)
        else:
            keypoint_sequences.append(np.zeros(TOTAL_KEYPOINT_DIM))

    # Pad or truncate sequences to the desired length
    if len(keypoint_sequences) < sequence_length:
        padding = [np.zeros(TOTAL_KEYPOINT_DIM)] * (sequence_length - len(keypoint_sequences))
        keypoint_sequences.extend(padding)
    elif len(keypoint_sequences) > sequence_length:
        keypoint_sequences = keypoint_sequences[:sequence_length]

    keypoint_sequences = torch.tensor(np.array(keypoint_sequences), dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        # 1. Pose-Based Feature Extraction
        tcn_output = model_tcn(keypoint_sequences)

        # 2. Visual Feature Extraction
        # (Replace with your actual frame loading/processing)
        batch_size = keypoint_sequences.size(0)
        num_frames = tcn_output.size(1)
        dummy_frames = torch.randn(batch_size * num_frames, 3, 224, 224).to(device) # Replace with actual frames
        visual_features = model_visual(dummy_frames)
        visual_features = visual_features.view(batch_size, num_frames, VISUAL_FEATURE_DIM)

        # 3. Concatenate Features
        combined_features = torch.cat((tcn_output, visual_features), dim=-1)

        # 4. MLSLT Encode
        shared_features = model_mlslt.encode(combined_features)

        # 5. Classification
        output = classifier(shared_features)
        predictions = torch.argmax(output, dim=1)

    # Map the predicted index to the sign word
    predicted_class_idx = predictions.item()
    predicted_sign = train_dataset.classes[predicted_class_idx]  # Assuming train_dataset.classes is accessible

    return predicted_sign

if __name__ == '__main__':
    # Example Usage (replace with your actual model loading and video path)
    from model.temporal_cnn import TemporalCNN
    from model.mlslt import MLSLT
    from model.visual_extractor import VisualFeatureExtractor
    from utils.config import *
    from dataloader.dataset import SignLanguageDataset

    # Load a dataset to get the class mapping
    train_dataset = SignLanguageDataset(root_dir=TRAIN_DIR, sequence_length=30)

    # Initialize Models
    model_tcn = TemporalCNN(input_size=TOTAL_KEYPOINT_DIM, hidden_size=256, num_layers=2).to(DEVICE)
    model_mlslt = MLSLT(encoder_input_dim=256 + VISUAL_FEATURE_DIM,
                         decoder_input_dim=256,
                         hidden_dim=MLSLT_HIDDEN_DIM,
                         num_layers=MLSLT_NUM_LAYERS,
                         output_dim=100,
                         dropout=MLSLT_DROPOUT).to(DEVICE)
    model_visual = VisualFeatureExtractor().to(DEVICE)
    classifier = nn.Linear(MLSLT_HIDDEN_DIM, NUM_CLASSES).to(DEVICE)

    # Load trained model weights
    # Replace with the actual path to your saved checkpoint
    checkpoint_path = "saved_models/checkpoint.pth"
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_tcn.load_state_dict(checkpoint['model_state_dict']['model_tcn'])
        model_mlslt.load_state_dict(checkpoint['model_state_dict']['model_mlslt'])
        model_visual.load_state_dict(checkpoint['model_state_dict']['model_visual'])
        classifier.load_state_dict(checkpoint['model_state_dict']['classifier'])
        print(f"Loaded model from {checkpoint_path}")
    elif:
        print(f"Checkpoint not found at {checkpoint_path}.initialized weights.")
    else:
        print("Using randomly initialized weights.")
    # Example Inference
    sample_video_path = "dataset/test/ISL_Auslan/word1/video1.mp4"  # Replace with a valid test video path
    predicted_sign = inference(sample_video_path, model_tcn, model_visual, model_mlslt, classifier, DEVICE)
    print(f"Predicted sign for {sample_video_path}: {predicted_sign}")