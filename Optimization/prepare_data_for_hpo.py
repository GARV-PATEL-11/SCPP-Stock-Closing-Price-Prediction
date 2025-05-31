# prepare_data_for_hpo.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from DataPipeline.data_loader import load_data  # Ensure this module has a load_data function
from Utils.get_sequenced_data import StockDataset  # Custom dataset class
from Optimization.select_features_for_model import select_features_for_model  # Function to select model-specific features

def prepare_data_for_hpo(Modelling_data, model_type, batch_size=128, seq_length=60):
    """
    Prepare data for hyperparameter optimization using time-series models.

    Parameters:
    -----------
    Modelling_data : str
        Filepath to the dataset (CSV, Excel, etc.).
    model_type : str
        Model architecture type ('lstm', 'cnn', etc.).
    batch_size : int, optional
        Samples per batch (default: 128).
    seq_length : int, optional
        Time steps per input sequence (default: 45).

    Returns:
    --------
    data : pd.DataFrame
        Raw loaded dataset.
    train_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader
        Validation DataLoader.
    scaler_X : MinMaxScaler
        Scaler for input features.
    scaler_y : StandardScaler
        Scaler for the target variable.
    input_features : list
        Selected feature names.
    num_features : int
        Number of input features.
    """

    # Load dataset
    data = load_data(Modelling_data)

    # Feature selection based on model type
    input_features = select_features_for_model(data, model_type)

    # Define split index
    len_data = len(data)
    train_len = int(len_data * 0.85 )
    val_len = len_data - train_len
    
    # Define split index
    train_data = data.iloc[:train_len].copy()
    val_data = data.iloc[train_len + 1:].copy()

    # Extract input and target variables
    X_train = train_data[input_features].values
    y_train = train_data['Close'].values.reshape(-1, 1)
    X_val = val_data[input_features].values
    y_val = val_data['Close'].values.reshape(-1, 1)

    # Normalize input features
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    # Normalize target variable
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Create datasets
    train_dataset = StockDataset(X_train_scaled, y_train_scaled, seq_length)
    val_dataset = StockDataset(X_val_scaled, y_val_scaled, seq_length)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Count input features
    num_features = len(input_features)

    return data, train_loader, val_loader, scaler_X, scaler_y, input_features, num_features
