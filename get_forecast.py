import pandas as pd
import torch
from Optimization.select_features_for_model import select_features_for_model

def generate_forecast(
    forecast_data: pd.DataFrame,
    Trained_Model: torch.nn.Module,
    X_scaler,
    y_scaler,
    seq_length: int,
    ticker: str,
    model: str
) -> None :
    """
    Generate a forecast for the next business day using a trained model.

    Parameters:
    - forecast_data (pd.DataFrame): Historical data including features.
    - Trained_Model (torch.nn.Module): Trained PyTorch model for prediction.
    - X_scaler: Scaler used to normalize the input features (e.g., StandardScaler).
    - y_scaler: Scaler used to normalize/invert the output target variable.
    - seq_length (int): Number of past time steps to use as input.
    - ticker (str): Ticker symbol for the stock (used for printing output).

    Returns:
    - float: Predicted close price for the next business day.
    """
    # --- Step 1: Feature Selection ---
    valid_features = select_features_for_model(forecast_data, model)

    # --- Step 2: Extract Input Window ---
    window_df = forecast_data[valid_features].iloc[-seq_length:]
    
    # --- Step 3: Handle Missing Values ---
    window_df = window_df.dropna()
    if len(window_df) < seq_length:
        raise ValueError("Insufficient data after dropping missing values.")

    # --- Step 4: Feature Scaling ---
    window_scaled = X_scaler.transform(window_df.values)

    # --- Step 5: Prepare Tensor Input ---
    x_in = torch.from_numpy(window_scaled).float().unsqueeze(0)

    # --- Step 6: Move Tensor to Model Device ---
    device = next(Trained_Model.parameters()).device
    x_in = x_in.to(device)

    # --- Step 7: Model Inference ---
    Trained_Model.eval()
    with torch.no_grad():
        y_scaled = Trained_Model(x_in)

    # --- Step 8: Invert Scaling of Prediction ---
    y_pred = y_scaler.inverse_transform(y_scaled.cpu().numpy())

    # --- Step 9: Determine Prediction Date ---
    last_date = forecast_data.index[-1]
    predicted_date = last_date + pd.offsets.BDay(1)

    # --- Step 10: Print and Return Prediction ---
    predicted_price = y_pred[0, 0]
    print(f"Predicted Close Price of {ticker} for {predicted_date.date()}: {predicted_price:.4f}")

