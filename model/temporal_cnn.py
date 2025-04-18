import torch
import torch.nn as nn
from utils.config import MOTION_FEATURE_DIM, TCN_OUTPUT_SEQUENCE_LENGTH

class TemporalCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_sequence_length=TCN_OUTPUT_SEQUENCE_LENGTH):
        super(TemporalCNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.projection = nn.Linear(hidden_size, MOTION_FEATURE_DIM) # Project to 256
        self.output_sequence_length = output_sequence_length

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # out shape: (batch_size, seq_len, hidden_size)

        out = self.max_pool(out.permute(0, 2, 1)).permute(0, 2, 1)
        # out shape: (batch_size, seq_len // 2, hidden_size) (if seq_len is even)

        out = self.projection(out)
        # out shape: (batch_size, seq_len // 2, 256)

        # Adjust sequence length to the desired output
        if out.shape[1] > self.output_sequence_length:
            out = out[:, :self.output_sequence_length, :]
        elif out.shape[1] < self.output_sequence_length:
            padding = torch.zeros(out.shape[0], self.output_sequence_length - out.shape[1], out.shape[2]).to(out.device)
            out = torch.cat((out, padding), dim=1)

        return out  # Shape: (batch_size, output_sequence_length, 256)

if __name__ == '__main__':
    # Example Usage
    input_size = 33 * 3 + 21 * 3 + 21 * 3 + 50 * 3
    hidden_size = 512
    num_layers = 2
    batch_size = 16
    seq_len = 30

    model = TemporalCNN(input_size, hidden_size, num_layers)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output = model(dummy_input)
    print(output.shape)  # Expected: [16, 10, 256]