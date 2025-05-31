# select_features_for_model.py

import pandas as pd

def select_features_for_model(data: pd.DataFrame, model_type: str) -> list:
    """
    Select appropriate features from the dataset for the specified model type.

    This function identifies a base set of features, removes 'Close' (commonly
    the target), validates their presence in the dataset, and optionally allows
    customization by model type.

    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset containing feature columns.
    model_type : str
        Type of model to be trained (e.g., 'lstm', 'cnn', etc.).
        This is used for future customization of feature sets per model type.

    Returns:
    --------
    input_features : list of str
        Final list of features present in the dataset to be used for training.
    """

    # Base feature set: lags, technical indicators, derivatives, volatility, etc.
    base_features = features = [
    'Close',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Low_Lag_1', 'Low_Lag_2', 'Low_Lag_3', 
    'High_Lag_1', 'High_Lag_2', 'High_Lag_3',
    'Open_Lag_1', 'Open_Lag_2', 'Open_Lag_3', 
    'Daily_Range_Lag_1', 'Daily_Range_Lag_2',
    'Daily_Range_Lag_3', 'Daily_Change_Lag_1',
    'Daily_Change_Lag_2', 'Daily_Change_Lag_3',
    'EMA_3', 'EMA_5', 'EMA_7','EMA_9', 'EMA_14', 'EMA_21',
    'ATR_3', 'ATR_5', 'ATR_7','ATR_9', 'ATR_14', 'ATR_21',
    'ROC_3', 'ROC_5', 'ROC_7', 'ROC_9', 'ROC_14', 'ROC_21',
    'Close_Lag_1_1th_Derivative', 'Close_Lag_1_2th_Derivative', 'Close_Lag_1_3th_Derivative',
    'Rolling_STD_2', 'Rolling_STD_3', 'Rolling_STD_5', 'Rolling_STD_7', 'Rolling_STD_9','Rolling_STD_14', 'Rolling_STD_21', 
    'ZScore_Close_2', 'ZScore_Close_3', 'ZScore_Close_5','ZScore_Close_7', 'ZScore_Close_9', 'ZScore_Close_14', 'ZScore_Close_21',
    'BB_Width_2', 'BB_Width_3', 'BB_Width_5', 'BB_Width_7', 'BB_Width_14', 'BB_Width_21'
]


    # Remove 'Close' to avoid data leakage as it's typically the main target
    input_features = [f for f in base_features if f != 'Close']

    # (Optional) Modify feature set based on model_type
    if model_type.lower() in ['lstm', 'bilstm','rnn']:
        # Future model-specific feature logic could go here
        pass

    # Filter features to only those present in the input DataFrame
    missing_features = [f for f in input_features if f not in data.columns]
    
    if missing_features:
        print(f"Warning: Missing features from dataset: {missing_features}")
        input_features = [f for f in input_features if f in data.columns]

    return input_features

