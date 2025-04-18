import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics(predictions, targets, average='weighted'):
    """
    Calculates evaluation metrics.

    Args:
        predictions (torch.Tensor): Predicted class indices.
        targets (torch.Tensor): Ground truth class indices.
        average (str): Averaging method for precision, recall, f1-score.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and f1-score.
    """

    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    accuracy = accuracy_score(targets_cpu, predictions_cpu)
    precision, recall, f1, _ = precision_recall_fscore_support(targets_cpu, predictions_cpu, average=average)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(model_tcn, model_visual, model_mlslt, classifier, data_loader, device):
    """
    Evaluates the model on the given data loader.

    Args:
        model_tcn (nn.Module): TemporalCNN model.
        model_visual (nn.Module): VisualFeatureExtractor model.
        model_mlslt (nn.Module): MLSLT model.
        classifier (nn.Module): SignClassifier (transfer head).
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to use for evaluation.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """

    model_tcn.eval()
    model_visual.eval()
    model_mlslt.eval()
    classifier.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (keypoint_sequences, labels) in enumerate(data_loader):
            keypoint_sequences = keypoint_sequences.to(device)
            labels = labels.to(device)

            # 1. Pose-Based Feature Extraction
            tcn_output = model_tcn(keypoint_sequences)

            # 2. Visual Feature Extraction
            # (Replace with your actual frame loading/processing)
            batch_size = keypoint_sequences.size(0)
            num_frames = keypoint_sequences.size(1)
            dummy_frames = torch.randn(batch_size * num_frames, 3, 224, 224).to(device)
            visual_features = model_visual(dummy_frames)
            visual_features = visual_features.view(batch_size, num_frames, VISUAL_FEATURE_DIM)

            # 3. Concatenate Features
            combined_features = torch.cat((tcn_output, visual_features), dim=-1)

            # 4. MLSLT Encode
            shared_features = model_mlslt.encode(combined_features)

            # 5. Classification
            output = classifier(shared_features)
            predictions = torch.argmax(output, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return calculate_metrics(torch.tensor(all_predictions), torch.tensor(all_targets))

if __name__ == '__main__':
    # Example Usage (replace with your actual model loading and data)
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy data
    dummy_predictions = torch.randint(0, 10, (100,))
    dummy_targets = torch.randint(0, 10, (100,))
    metrics = calculate_metrics(dummy_predictions, dummy_targets)
    print("Example Metrics:", metrics)

    # Example Evaluation
    class DummyModel(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.fc = nn.Linear(10, output_size)

        def forward(self, x):
            return self.fc(x)

    dummy_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 10, (100,)))
    dummy_loader = DataLoader(dummy_data, batch_size=10)
    dummy_model = DummyModel(10)
    #dummy_metrics = evaluate_model(dummy_model, dummy_loader, torch.device('cpu'))
    #print("Example Evaluation Metrics:", dummy_metrics)