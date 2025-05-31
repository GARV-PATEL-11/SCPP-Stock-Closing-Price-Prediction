# create_model_with_params.py

import torch
from Models.model_architectures import LSTMModel
from Models.model_architectures import BiLSTMModel
from Models.model_architectures import RNNModel

def create_model_with_params(model_type, num_features, params):
    """
    Create a time series model with the specified architecture and parameters.

    Args:
        model_type (str): Type of model (e.g., 'lstm', 'gru', 'cnn_lstm', etc.)
        num_features (int): Number of input features for the model.
        params (dict): Dictionary of model parameters.

    Returns:
        torch.nn.Module: An instance of the requested model.
    """
    if model_type == 'lstm':
        return LSTMModel(
            input_size=num_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            cell_dropout=params.get('cell_dropout', 0.0),
            batch_norm=params.get('batch_norm', False)
        )

    elif model_type == 'bilstm':
        return BiLSTMModel(
            input_size=num_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            cell_dropout=params.get('cell_dropout', 0.0),
            batch_norm=params.get('batch_norm', False)
        )

    elif model_type == 'rnn':
        return RNNModel(
            input_size=num_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            nonlinearity=params.get('nonlinearity', 'tanh'),
            batch_norm=params.get('batch_norm', False)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
