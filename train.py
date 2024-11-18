import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharTransformerModel
import numpy as np
from torch.cuda import amp
import random

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Model and training parameters
embed_size = 256
num_heads = 8
hidden_dim = 512
num_layers = 4
dropout = 0.1
batch_size = 256
seq_length = 128
num_epochs = 50
learning_rate = 1e-4

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build character vocabulary without loading the entire dataset into RAM
def build_vocab(dataset_path):
    chars = set()
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(dataset_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(1024 * 1024)  # Read in 1MB chunks
                    if not chunk:
                        break
                    chars.update(chunk)
    chars = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, chars

stoi, itos, chars = build_vocab('dataset')
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# TextDataset class that reads data on-the-fly
class TextDataset(Dataset):
    def __init__(self, dataset_path, stoi, seq_length):
        self.file_paths = [os.path.join(dataset_path, filename)
                           for filename in os.listdir(dataset_path) if filename.endswith(".txt")]
        self.stoi = stoi
        self.seq_length = seq_length
        self.files_data = []

        # Calculate total sequences across all files
        self.total_sequences = 0
        self.file_sequences = []
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
            num_sequences = max(0, file_size - self.seq_length - 1)
            self.file_sequences.append(num_sequences)
            self.total_sequences += num_sequences

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Determine which file the idx falls into
        cumulative_sequences = 0
        for file_idx, num_sequences in enumerate(self.file_sequences):
            if cumulative_sequences + num_sequences > idx:
                break
            cumulative_sequences += num_sequences

        # Adjust idx to the local index within the file
        local_idx = idx - cumulative_sequences
        file_path = self.file_paths[file_idx]

        with open(file_path, 'r', encoding='utf-8') as f:
            # Seek to the starting position
            f.seek(local_idx)
            data = f.read(self.seq_length + 1)
            # If not enough data, pad with spaces
            if len(data) < self.seq_length + 1:
                data += ' ' * (self.seq_length + 1 - len(data))
            data_indices = [self.stoi.get(ch, self.stoi[' ']) for ch in data]
            input_seq = torch.tensor(data_indices[:-1], dtype=torch.long)
            target_seq = torch.tensor(data_indices[1:], dtype=torch.long)
            return input_seq, target_seq

# Create the DataLoader
dataset = TextDataset('dataset', stoi, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, pin_memory=True, num_workers=4)

# Initialize the model
model = CharTransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Use mixed precision training
scaler = amp.GradScaler()

# Training function
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.transpose(0, 1).to(device, non_blocking=True)
            target_seq = target_seq.transpose(0, 1).to(device, non_blocking=True)

            optimizer.zero_grad()
            with amp.autocast():
                output = model(input_seq)
                loss = criterion(output.view(-1, vocab_size), target_seq.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = f'checkpoints/model_epoch_{epoch + 1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'stoi': stoi,
            'itos': itos,
            'embed_size': embed_size,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save the final model
    torch.save(model.state_dict(), 'model_final.pth')

# Start training
if __name__ == '__main__':
    train(model, dataloader, criterion, optimizer, num_epochs)