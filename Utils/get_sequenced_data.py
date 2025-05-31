import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1  # ✅ include last valid sequence

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_length]
        y_target = self.y[idx + self.seq_length - 1]
        return X_seq, y_target
