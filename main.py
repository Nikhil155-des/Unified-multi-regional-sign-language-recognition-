import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader.dataset import SignLanguageDataset
from model.temporal_cnn import TemporalCNN
from model.mlslt import MLSLT
from model.visual_extractor import VisualFeatureExtractor
from utils.config import * # Import all configs

class SignClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SignClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # The Transfer Head

    def forward(self, x):
        return self.fc(x)

def train_one_epoch(model_tcn, model_mlslt, model_visual, classifier, data_loader, optimizer, device):
    model_tcn.train()
    model_mlslt.train()
    model_visual.train()
    classifier.train() # Set classifier to train mode
    total_loss = 0
    for batch_idx, (keypoint_sequences, labels) in enumerate(data_loader):
        keypoint_sequences = keypoint_sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 1. Pose-Based Feature Extraction
        tcn_output = model_tcn(keypoint_sequences)  # (batch_size, 10, 256)

        # 2. Visual Feature Extraction
        # Assuming your dataset returns frames along with keypoints, or you load them here
        # For simplicity, let's create dummy visual features (replace with actual frame loading)
        batch_size = keypoint_sequences.size(0)
        num_frames = keypoint_sequences.size(1)
        dummy_frames = torch.randn(batch_size * num_frames, 3, 224, 224).to(device)  # (batch_size * 10, 3, 224, 224)
        visual_features = model_visual(dummy_frames)  # (batch_size * 10, 512)
        visual_features = visual_features.view(batch_size, num_frames, VISUAL_FEATURE_DIM)  # (batch_size, 10, 512)

        # 3. Concatenate (or Fuse) Pose and Visual Features
        # Example: Simple concatenation
        combined_features = torch.cat((tcn_output, visual_features), dim=-1)  # (batch_size, 10, 256 + 512)

        # 4. MLSLT Encoder
        # Use only the encoder to get the shared representation
        shared_features = model_mlslt.encode(combined_features)  # (batch_size, MLSLT_HIDDEN_DIM)

        # 5. Classification (using the Transfer Head)
        output = classifier(shared_features)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def main():
    # Load Data
    train_dataset = SignLanguageDataset(root_dir=TRAIN_DIR, sequence_length=30)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize Models
    model_tcn = TemporalCNN(input_size=TOTAL_KEYPOINT_DIM, hidden_size=256, num_layers=2).to(DEVICE)
    model_mlslt = MLSLT(encoder_input_dim=256 + VISUAL_FEATURE_DIM,  # Input dim to MLSLT encoder
                         decoder_input_dim=256,  # Example
                         hidden_dim=MLSLT_HIDDEN_DIM,
                         num_layers=MLSLT_NUM_LAYERS,
                         output_dim=100,  # Example
                         dropout=MLSLT_DROPOUT).to(DEVICE)
    model_visual = VisualFeatureExtractor().to(DEVICE)

    # Initialize Transfer Head (SignClassifier)
    classifier = SignClassifier(input_dim=MLSLT_HIDDEN_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    # Optimizer
    optimizer = Adam(list(model_tcn.parameters()) +
                     list(model_mlslt.parameters()) +
                     list(model_visual.parameters()) +
                     list(classifier.parameters()), # Include classifier parameters
                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model_tcn, model_mlslt, model_visual, classifier, train_dataloader, optimizer, DEVICE)
        print(f'Epoch: {epoch + 1}, Loss: {avg_loss}')

if __name__ == '__main__':
    main()