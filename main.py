import argparse
import torch
import torch.nn as nn
import numpy as np
from utils.config import *
from train import train
from model.temporal_cnn import TemporalCNN
from model.visual_extractor import VisualFeatureExtractor
from model.mlslt import MLSLT
from dataloader.dataset import SignLanguageDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import evaluate_model

class SignClassifier(nn.Module):
    """Final classification head for sign language recognition."""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SignClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Multi-Regional Sign Language Recognition")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'],
                       help="Mode to run: train, test, or inference")
    parser.add_argument('--checkpoint', type=str, default='saved_models/final_model.pth',
                       help="Path to checkpoint for testing or inference")
    parser.add_argument('--video_path', type=str, default=None,
                       help="Path to video file for inference mode")
    return parser.parse_args()

def test_model(checkpoint_path):
    device = torch.device(DEVICE)
    
    model_tcn = TemporalCNN(TOTAL_KEYPOINT_DIM, hidden_size=512, num_layers=2).to(device)
    model_visual = VisualFeatureExtractor().to(device)
    model_mlslt = MLSLT(
        encoder_input_dim=MOTION_FEATURE_DIM + VISUAL_FEATURE_DIM,
        decoder_input_dim=512,
        hidden_dim=MLSLT_HIDDEN_DIM,
        num_layers=MLSLT_NUM_LAYERS,
        output_dim=100,
        dropout=MLSLT_DROPOUT
    ).to(device)
    classifier = SignClassifier(MLSLT_HIDDEN_DIM, 256, NUM_CLASSES).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    model_tcn.load_state_dict(checkpoint['model_tcn'])
    model_visual.load_state_dict(checkpoint['model_visual'])
    model_mlslt.load_state_dict(checkpoint['model_mlslt'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    test_dataset = SignLanguageDataset(TEST_DIR, sequence_length=MAX_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    metrics = evaluate_model(model_tcn, model_visual, model_mlslt, classifier, test_loader, device)
    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

def inference(checkpoint_path, video_path):
    device = torch.device(DEVICE)
    
    model_tcn = TemporalCNN(TOTAL_KEYPOINT_DIM, hidden_size=512, num_layers=2).to(device)
    model_visual = VisualFeatureExtractor().to(device)
    model_mlslt = MLSLT(
        encoder_input_dim=MOTION_FEATURE_DIM + VISUAL_FEATURE_DIM,
        decoder_input_dim=512,
        hidden_dim=MLSLT_HIDDEN_DIM,
        num_layers=MLSLT_NUM_LAYERS,
        output_dim=100,
        dropout=MLSLT_DROPOUT
    ).to(device)
    classifier = SignClassifier(MLSLT_HIDDEN_DIM, 256, NUM_CLASSES).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    model_tcn.load_state_dict(checkpoint['model_tcn'])
    model_visual.load_state_dict(checkpoint['model_visual'])
    model_mlslt.load_state_dict(checkpoint['model_mlslt'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    from dataloader.video_preprocessing import process_video, sample_frames
    from dataloader.pose_estimation import PoseEstimator
    
    frames = process_video(video_path)
    frames = sample_frames(frames, max_frames=MAX_FRAMES)
    
    pose_estimator = PoseEstimator()
    keypoint_sequences = []
    for frame in frames:
        keypoints = pose_estimator.extract_keypoints(frame)
        keypoint_sequences.append(keypoints if keypoints is not None else np.zeros(TOTAL_KEYPOINT_DIM))
     
    if len(keypoint_sequences) < MAX_FRAMES:
        keypoint_sequences.extend([np.zeros(TOTAL_KEYPOINT_DIM)] * (MAX_FRAMES - len(keypoint_sequences)))
    elif len(keypoint_sequences) > MAX_FRAMES:
        keypoint_sequences = keypoint_sequences[:MAX_FRAMES]
    
    keypoint_sequences = torch.tensor(np.array(keypoint_sequences), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        model_tcn.eval()
        model_visual.eval()
        model_mlslt.eval()
        classifier.eval()
        
        tcn_output = model_tcn(keypoint_sequences)
        dummy_frames = torch.randn(1 * MAX_FRAMES, 3, 224, 224).to(device)
        visual_features = model_visual(dummy_frames)
        visual_features = visual_features.view(1, MAX_FRAMES, VISUAL_FEATURE_DIM)
        combined_features = torch.cat((tcn_output, visual_features), dim=-1)
        shared_features = model_mlslt.encode(combined_features)
        outputs = classifier(shared_features)
        _, predicted = torch.max(outputs.data, 1)
    
    test_dataset = SignLanguageDataset(TEST_DIR)
    predicted_label = test_dataset.classes[predicted.item()]
    print(f"\nPredicted Sign: {predicted_label}")

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_model(args.checkpoint)
    elif args.mode == 'inference':
        if not args.video_path:
            raise ValueError("Video path is required for inference")
        inference(args.checkpoint, args.video_path)