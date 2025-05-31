# get_models.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, cell_dropout=0.0, batch_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(cell_dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        if self.batch_norm:
            x = x.view(-1, features)
            x = self.bn(x)
            x = x.view(batch_size, seq_len, features)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, cell_dropout=0.0, batch_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(cell_dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        if self.batch_norm:
            x = x.view(-1, features)
            x = self.bn(x)
            x = x.view(batch_size, seq_len, features)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class RNNModel(nn.Module):
    """Simple RNN model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, nonlinearity='tanh', batch_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity=nonlinearity
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        if self.batch_norm:
            x = x.view(-1, features)
            x = self.bn(x)
            x = x.view(batch_size, seq_len, features)

        rnn_out, _ = self.rnn(x)
        out = rnn_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
