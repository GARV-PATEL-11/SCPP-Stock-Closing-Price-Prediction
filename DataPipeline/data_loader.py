import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple

def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and preprocess stock data.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and formatted DataFrame.
    """
    data = df.copy()

    # Convert 'Date' column to datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Drop unnamed column if present (e.g., from CSV index)
    if data.columns[-1].startswith('Unnamed'):
        data.drop(columns=data.columns[-1], inplace=True)

    return data
