import torch
import torch.nn as nn
from utils.config import MLSLT_FEATURE_DIM, MLSLT_HIDDEN_DIM, MLSLT_NUM_LAYERS, MLSLT_DROPOUT

class MLSLTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MLSLTEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) - Output from TemporalCNN (e.g., seq_len=10, input_dim=256)
        outputs, hidden = self.gru(x)
        return outputs, hidden # outputs: (batch_size, seq_len, hidden_dim), hidden: (num_layers, batch_size, hidden_dim)

class MLSLTDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(MLSLTDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, input_dim) # Assuming input_dim can be embedding dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # x shape: (batch_size, 1) - single word index
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, 1, input_dim)
        output, hidden = self.gru(embedded, hidden)
        # output shape: (batch_size, 1, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))
        # prediction shape: (batch_size, output_dim)
        return prediction, hidden

class MLSLT(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(MLSLT, self).__init__()
        self.encoder = MLSLTEncoder(encoder_input_dim, hidden_dim, num_layers, dropout)
        self.decoder = MLSLTDecoder(decoder_input_dim, hidden_dim, output_dim, num_layers, dropout)

    def forward(self, src, trg, teacher_force_ratio=0.5):
        # src shape: (batch_size, src_len, encoder_input_dim) - Output from TemporalCNN
        # trg shape: (batch_size, trg_len) - Target word indices

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)

        # Use the final hidden state of the encoder as the initial hidden state of the decoder
        decoder_hidden = hidden

        # First input to the decoder is the <SOS> token
        input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, decoder_hidden) # Use decoder_hidden here
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_force_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1
            decoder_hidden = hidden # Update decoder hidden state

        return outputs

    def encode(self, src):
        """Only use the encoder to get the shared representation."""
        encoder_outputs, encoder_hidden = self.encoder(src)
        # We might want to return the last hidden state or a pooled representation
        # For simplicity, let's return the last hidden state of the last layer
        return encoder_hidden[-1]  # Shape: (batch_size, hidden_dim)

if __name__ == '__main__':
    # Example Usage
    encoder_input_dim = 256 # From TCN
    decoder_input_dim = 256 # Can be embedding dim
    hidden_dim = 512
    num_layers = 2
    output_dim = 100 # Vocabulary size
    dropout = 0.1
    batch_size = 16
    seq_len = 10
    trg_len = 5

    model = MLSLT(encoder_input_dim, decoder_input_dim, hidden_dim, num_layers, output_dim, dropout).to('cuda')
    dummy_src = torch.randn(batch_size, seq_len, encoder_input_dim).to('cuda')
    dummy_trg = torch.randint(0, output_dim, (batch_size, trg_len)).to('cuda')

    output = model(dummy_src, dummy_trg)
    print(f"MLSLT output shape: {output.shape}") # Should be (trg_len, batch_size, output_dim)

    # Example of using the encode function
    encoder_output = model.encode(dummy_src)
    print(f"MLSLT Encoder output shape: {encoder_output.shape}") # Should be (batch_size, hidden_dim)