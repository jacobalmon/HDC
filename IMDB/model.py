import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import random

# Set Random Seeds.
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define HDC Class.
class HDC:
    def __init__(self, dim):
        self.dim = dim  # Dimensionality of the hypervectors.

    # Generate a Random Hypervector (-1 to 1).
    def random_hv(self):
        hv = np.random.randint(2, size=self.dim)  # Binary hypervector with values 0 or 1.
        return hv

    # Element-wise Addition and Normalization.
    def superpose(self, hvs):
        sum_hv = np.sum(hvs, axis=0)  # Sum all input hypervectors element-wise.
        norm = np.linalg.norm(sum_hv)  # Compute L2 norm of the summed hypervector.
        if norm == 0:
            return sum_hv  # Avoid division by zero; return as is.
        return sum_hv / norm  # Return normalized hypervector.

    # Element-wise Multiplication.
    def bind(self, hv1, hv2):
        return hv1 * hv2  # Hadamard product (element-wise multiplication).

    # Circular Shift for Encoding Position.
    def permute(self, hv):
        return np.roll(hv, 1)  # Shift elements by 1 to the right (circularly).
    
# Dataset Class for IMDb
class IMDbDataset(Dataset):
    def __init__(self, data, vocab, token_hvs, hd, max_seq_len):
        self.data = data  # IMDb dataset (list of dicts with 'text' and 'label')
        self.vocab = vocab  # Vocabulary set
        self.token_hvs = token_hvs  # Dictionary mapping tokens to hypervectors
        self.hd = hd  # HDC (Hyperdimensional Computing) instance
        self.max_seq_len = max_seq_len  # Maximum sequence length to consider

    def __len__(self):
        return len(self.data)  # Number of data samples

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # Tokenize text: lowercase, split, and clip to max_seq_len
        tokens = text.lower().split()[:self.max_seq_len]
        tokens = ['[CLS]'] + tokens + ['[SEP]']  # Add special tokens

        # Encode tokens into a single hypervector representation
        seq_hv = encode_sequence(tokens, self.token_hvs, self.hd)

        # Convert label to tensor (binary classification: 1 or 0)
        label_tensor = torch.tensor(1.0 if label == 1 else 0.0, dtype=torch.float32)

        return torch.tensor(seq_hv, dtype=torch.float32), label_tensor

# Create Token Hypervectors.
def create_token_hvs(vocab, dim, hd):
    token_hvs = {}
    for token in vocab:
        token_hvs[token] = hd.random_hv()  # Assign random hypervector to each token
    return token_hvs

# Encode Sequences into Hypervectors.
def encode_sequence(tokens, token_hvs, hd):
    position_hv = np.ones(hd.dim)  # Start with a position vector (identity)
    sequence_hv = []
    for token in tokens:
        token_hv = token_hvs.get(token, hd.random_hv())  # Use existing or new token hypervector
        combined_hv = hd.bind(token_hv, position_hv)  # Bind token and position
        sequence_hv.append(combined_hv)
        position_hv = hd.permute(position_hv)  # Shift position for next token
    sequence_representation = hd.superpose(sequence_hv)  # Superpose all token-position bindings
    return sequence_representation

# HDC NN Model.
class HDCNN(nn.Module):
    def __init__(self, dim):
        super(HDCNN, self).__init__()
        self.dim = dim # Dimension of the input hypervector.

        # Simple Neural Network.
        self.fc1 = nn.Linear(dim, 2048)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(2048, dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x) # Linear Transformation.
        out = self.activation(out) # Apply ReLU.
        out = self.dropout(out) # Apply Dropout.
        out = self.fc2(out) # Project back to Original Dimension.

        # Normalize Output to Unit Normalization (L2 Norm).
        norm = out.norm(p=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm) # Avoid Divison by Zero.
        out = out / norm

        # Normalized Output Vector.
        return out

def main():
    # Initialize Parameters.
    dim = 5000
    hd = HDC(dim)
    max_vocab_size = 5000
    max_seq_len = 200
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    # Load IMDb Dataset.
    dataset = load_dataset('csv', data_files='data/IMDB%20Dataset.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2)
    dataset = dataset.rename_column("review", "text")
    dataset = dataset.rename_column("sentiment", "label")
    train_data = dataset['train']
    test_data = dataset['test']

    # Build Vocabulary.
    counter = Counter()
    for item in train_data:
        tokens = item['text'].lower().split()
        counter.update(tokens)
    vocab_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + [token for token, _ in counter.most_common(max_vocab_size)]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    # Create Token Hypervectors.
    token_hvs = create_token_hvs(vocab, dim, hd)

    # Create Datasets.
    train_data = IMDbDataset(train_data, vocab, token_hvs, hd, max_seq_len)
    test_data = IMDbDataset(test_data, vocab, token_hvs, hd, max_seq_len)

    # Create Data Loaders.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss Function, and Optimizer.
    model = HDCNN(dim)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Model to GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train Model.
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            target = torch.ones(inputs.size(0)).to(device)
            loss = criterion(outputs, inputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')\

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
        
        # Evaluate Model.
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                target = torch.ones(inputs.size(0)).to(device)
                loss = criterion(outputs, inputs, target)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.6f}')
      
        model.eval()
        with torch.no_grad():
            similarities = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                sim = torch.cosine_similarity(inputs, outputs, dim=1)
                similarities.extend(sim.tolist())

            avg_similarity = sum(similarities) / len(similarities)
            print(f'Average Cosine Similarity on Test Set: {avg_similarity:.4f}')
            
if __name__ == '__main__':
    main()