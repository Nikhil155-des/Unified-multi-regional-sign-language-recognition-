import torch
import os

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """
    Saves a model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch number.
        filename (str): The filename for the checkpoint.
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """
    Loads a model checkpoint.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): The filename of the checkpoint.

    Returns:
        int: The epoch number from the checkpoint, or 0 if loading fails.
    """

    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    else:
        print(f"Checkpoint not found at {filename}")
        return 0

if __name__ == '__main__':
    # Example Usage
    import torch.nn as nn
    import torch.optim as optim

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Save a checkpoint
    save_checkpoint(model, optimizer, epoch=5, filename="test_checkpoint.pth")

    # Load the checkpoint
    loaded_model = DummyModel()
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.01)
    loaded_epoch = load_checkpoint(loaded_model, loaded_optimizer, filename="test_checkpoint.pth")

    print("Loaded epoch:", loaded_epoch)