import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error

def evaluate_model(model, data_loader, train_loader=None):
    """
    Evaluates a PyTorch model and prints detailed metrics.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for evaluation (validation/test set).
        train_loader: Optional training DataLoader (used if data_loader is None).

    Returns:
        Dictionary of evaluation metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Use train_loader if data_loader is not provided
    if data_loader is None:
        if train_loader is not None:
            print("[WARNING] No validation/test DataLoader provided. Using train_loader for evaluation.")
            data_loader = train_loader
        else:
            raise ValueError("Both data_loader and train_loader are None. Cannot perform evaluation.")

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X_batch, y_batch = batch
            else:
                raise ValueError("Expected each batch to be a tuple/list of (X, y)")

            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_pred.append(y_pred.cpu().numpy())
            all_true.append(y_batch.numpy())

    y_pred = np.vstack(all_pred)
    y_true = np.vstack(all_true)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    # MAPE calculation with masking
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else float('nan')

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'explained_variance': explained_var,
        'max_error': max_err
    }

    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper():<20}: {v:.6f}")

    return metrics