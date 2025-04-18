import torch
import torch.nn as nn
import torchvision.models as models
from utils.config import VISUAL_FEATURE_DIM

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='mobilenet_v3_small', pretrained=True):
        super(VisualFeatureExtractor, self).__init__()
        if model_name.startswith('mobilenet_v3'):
            self.model = models.mobilenet_v3_small(pretrained=pretrained) # or _large
            # Modify the classifier to output the desired feature dimension
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, VISUAL_FEATURE_DIM)
        elif model_name.startswith('resnet'):
            self.model = models.resnet18(pretrained=pretrained) # Or a larger ResNet
            self.model.fc = nn.Linear(self.model.fc.in_features, VISUAL_FEATURE_DIM)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Example Usage
    batch_size = 16
    num_channels = 3
    height = 224
    width = 224

    dummy_input = torch.randn(batch_size, num_channels, height, width).to('cuda')
    model = VisualFeatureExtractor(model_name='mobilenet_v3_small').to('cuda')
    output = model(dummy_input)
    print(f"Visual Feature Extractor output shape: {output.shape}") # Should be (batch_size, 512)
