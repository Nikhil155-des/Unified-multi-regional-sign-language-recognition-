import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import sys
import numpy as np

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.config import *
from utils.train_utils import save_checkpoint, load_checkpoint
from utils.eval_metrics import evaluate_model
from dataloader.dataset import SignLanguageDataset
from model.temporal_cnn import TemporalCNN
from model.visual_extractor import VisualFeatureExtractor
from model.mlslt import MLSLT

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

def train_epoch(model_tcn, model_visual, model_mlslt, classifier, 
                dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model_tcn.train()
    model_visual.train()
    model_mlslt.train()
    classifier.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (keypoint_sequences, labels) in enumerate(tqdm(dataloader, desc="Training")):
        keypoint_sequences = keypoint_sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        tcn_output = model_tcn(keypoint_sequences)
        
        batch_size = keypoint_sequences.size(0)
        num_frames = keypoint_sequences.size(1)
        dummy_frames = torch.randn(batch_size * num_frames, 3, 224, 224).to(device)
        visual_features = model_visual(dummy_frames)
        visual_features = visual_features.view(batch_size, num_frames, VISUAL_FEATURE_DIM)
        
        combined_features = torch.cat((tcn_output, visual_features), dim=-1)
        shared_features = model_mlslt.encode(combined_features)
        outputs = classifier(shared_features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model_tcn, model_visual, model_mlslt, classifier, dataloader, criterion, device):
    metrics = evaluate_model(model_tcn, model_visual, model_mlslt, classifier, dataloader, device)
    return metrics['f1'], metrics['accuracy']

def train():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Initialize models
    input_size = TOTAL_KEYPOINT_DIM
    model_tcn = TemporalCNN(input_size, hidden_size=512, num_layers=2).to(device)
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
    
    # Initialize dataset and dataloader
    train_dataset = SignLanguageDataset(TRAIN_DIR, sequence_length=MAX_FRAMES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    test_dataset = SignLanguageDataset(TEST_DIR, sequence_length=MAX_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model_tcn.parameters()) + 
        list(model_visual.parameters()) + 
        list(model_mlslt.parameters()) + 
        list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    best_f1 = 0.0
    early_stop_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model_tcn, model_visual, model_mlslt, classifier,
            train_loader, optimizer, criterion, device
        )
        
        val_f1, val_acc = validate(
            model_tcn, model_visual, model_mlslt, classifier,
            test_loader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_tcn': model_tcn.state_dict(),
                'model_visual': model_visual.state_dict(),
                'model_mlslt': model_mlslt.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_f1
            }, filename=os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch + 1} epochs!")
                break
        
        # Save periodic checkpoints
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_tcn': model_tcn.state_dict(),
                'model_visual': model_visual.state_dict(),
                'model_mlslt': model_mlslt.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_f1
            }, filename=os.path.join(MODEL_SAVE_DIR, f'epoch_{epoch+1}_model.pth'))
        
        scheduler.step(val_f1)
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_tcn': model_tcn.state_dict(),
        'model_visual': model_visual.state_dict(),
        'model_mlslt': model_mlslt.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_f1': best_f1
    }
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')
    torch.save(final_checkpoint, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    train()