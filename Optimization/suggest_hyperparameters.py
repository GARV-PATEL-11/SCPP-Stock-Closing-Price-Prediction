# suggest_hyperparameters.py

import optuna

def suggest_hyperparameters(trial, model_type):
    """
    Suggest hyperparameters for neural network models using Optuna.

    This function generates a dictionary of hyperparameters dynamically
    based on the model type, enabling use with various architectures
    like LSTM, GRU, CNN-LSTM, etc. It includes core model settings,
    optimizer choices, scheduler configurations, and loss functions.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        The Optuna trial object used for hyperparameter suggestion.
    model_type : str
        Type of model architecture (e.g., 'lstm', 'gru', 'cnn-lstm').

    Returns:
    --------
    dict
        Dictionary of suggested hyperparameters.
    """
    
    # Common hyperparameters
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 1024),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
    }

    # Model-specific hyperparameters
    model_type = model_type.lower()

    if model_type in ['lstm', 'bilstm', 'bi-lstm']:
        params['cell_dropout'] = trial.suggest_float('cell_dropout', 0.0, 0.5)

    if model_type in ['cnnlstm', 'cnnbilstm', 'cnn-bilstm', 'cnnrnn']:
        params['kernel_size'] = trial.suggest_int('kernel_size', 2, 5)
        params['cnn_channels'] = trial.suggest_int('cnn_channels', 16, 128)
        params['cnn_layers'] = trial.suggest_int('cnn_layers', 1, 3)

    if model_type in ['rnn', 'rnnlstm', 'rnngru', 'cnnrnn', 'rnnbilstm']:
        params['nonlinearity'] = trial.suggest_categorical('nonlinearity', ['tanh', 'relu'])

    # Optimizer
    params['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'radam', 'sgd'])

    # Learning rate scheduler
    params['lr_scheduler'] = trial.suggest_categorical('lr_scheduler', ['none', 'step', 'cosine', 'plateau'])

    if params['lr_scheduler'] == 'step':
        params['lr_step_size'] = trial.suggest_int('lr_step_size', 5, 30)
        params['lr_gamma'] = trial.suggest_float('lr_gamma', 0.1, 0.9)
    elif params['lr_scheduler'] == 'cosine':
        params['lr_T_max'] = trial.suggest_int('lr_T_max', 5, 50)
    elif params['lr_scheduler'] == 'plateau':
        params['lr_factor'] = trial.suggest_float('lr_factor', 0.1, 0.9)
        params['lr_patience'] = trial.suggest_int('lr_patience', 3, 15)

    # Loss function
    params['loss_function'] = trial.suggest_categorical(
        'loss_function',
        ['mse', 'mae', 'huber']
    )

    if params['loss_function'] == 'huber':
        params['delta'] = trial.suggest_float('delta', 0.1, 1.0)
    return params

