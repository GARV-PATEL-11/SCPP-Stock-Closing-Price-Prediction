import os
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from Models.create_model_with_params import create_model_with_params
from Models.train_model_with_best_checkpoints import train_model_with_best_checkpoints
from Models.evaluate_model import evaluate_model
from PlotScripts.get_time_series_comparison_plot import time_series_comparison_plot
from PlotScripts.get_scatter_plot import scatter_plot
import json


def train_final_model(best_params, model_type, train_loader, val_loader, num_features, 
                     scaler_y, input_features, data, output_dir = "Results"):
    """
    Train final model with best hyperparameters and evaluate performance.
    Stores best models during training, retrieves the best model after training,
    and plots only epochs where best models were achieved.
    
    Args:
        best_params: Dictionary of best hyperparameters
        model_type: Type of model
        train_loader: Training data loader
        val_loader: Validation data loader (can be None for training-only scenarios)
        num_features: Number of input features
        scaler_y: Scaler for target variable
        input_features: List of input feature names
        data: Original data DataFrame
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTraining final model with best hyperparameters...\n",best_params)
    
    # Create model with best parameters
    model = create_model_with_params(model_type, num_features, best_params)

    # Train model with longer patience and get best model checkpoints
    trained_model, history, best_model_metrics = train_model_with_best_checkpoints(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_type=model_type,
    best_params=best_params,
    output_dir="Results",
    num_epochs=500,
    patience=500
    )

    # Evaluate model on validation set (or training set if no validation data)
    val_metrics = evaluate_model(trained_model, data_loader=val_loader, train_loader=train_loader)
    
    print(f"\nFinal {model_type} Model Performance:")
    print(f"Validation RMSE: {val_metrics['rmse']:.8f}")
    print(f"Validation MAE: {val_metrics['mae']:.8f}")
    print(f"Validation R²: {val_metrics['r2']:.8f}")
    print(f"Validation MAPE: {val_metrics['mape']:.4f}%")
    
    # Save training history plot using Plotly instead of matplotlib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subplot figure with Plotly
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'{model_type} Model Training History',
            'Learning Rate Schedule'
        )
    )
    
    # Plot 1: Training and validation loss
    fig.add_trace(
        go.Scatter(
            x=history['epochs'], 
            y=history['train_loss'], 
            mode='lines', 
            name='Train Loss',
            line=dict(color='blue', width=1),
            hovertemplate='Epoch: %{x}<br>Train Loss: %{y:.8f}',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history['epochs'], 
            y=history['val_loss'], 
            mode='lines', 
            name='Validation Loss',
            line=dict(color='red', width=1),
            hovertemplate='Epoch: %{x}<br>Val Loss: %{y:.8f}',
        ),
        row=1, col=1
    )
    
    # Highlight best epochs
    best_epochs = history['best_epoch_indices']
    best_val_losses = history['best_val_losses']
    best_train_losses = [history['train_loss'][history['epochs'].index(epoch)] for epoch in best_epochs]
    
    fig.add_trace(
        go.Scatter(
            x=best_epochs, 
            y=best_train_losses, 
            mode='markers', 
            name='Best Train Loss',
            marker=dict(color='green', size=10, symbol='circle'),
            hovertemplate='Best Epoch: %{x}<br>Train Loss: %{y:.8f}',
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=best_epochs, 
            y=best_val_losses, 
            mode='markers', 
            name='Best Val Loss',
            marker=dict(color='purple', size=12, symbol='star'),
            hovertemplate='Best Epoch: %{x}<br>Val Loss: %{y:.8f}',
        ),
        row=1, col=1
    )
    
    # Plot 2: Learning rate schedule
    fig.add_trace(
        go.Scatter(
            x=history['epochs'], 
            y=history['learning_rates'], 
            mode='lines', 
            name='Learning Rate',
            line=dict(color='orange', width=2),
            hovertemplate='Epoch: %{x}<br>LR: %{y:.8f}',
        ),
        row=1, col=2
    )  
    # Update axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
    
    # Generate predictions using the best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.to(device)
    trained_model.eval()
    
    # If no validation loader is provided, use the training loader for predictions
    prediction_loader = val_loader if val_loader is not None else train_loader
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in prediction_loader:
            X_batch = X_batch.to(device)
            y_pred = trained_model(X_batch)
            # Convert to numpy and store
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Inverse transform to original scale
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
 
    # Generate time series comparison
    time_series_comparison_plot(targets_original, predictions_original, model_type, output_dir,phase = "Validation")
    
    # Generate scatter plot analysis
    scatter_plot(targets_original, predictions_original, model_type, output_dir,phase = "Validation")
    
        
    # Save best model metrics to a file
    with open(os.path.join(output_dir, f"{model_type}_best_models_log_{timestamp}.json"), 'w') as f:
        json.dump([{
            'epoch': m['epoch'],
            'val_loss': m['val_loss'],
            'train_loss': m['train_loss']
        } for m in best_model_metrics], f, indent=2)

    # Save final best model
    model_file = os.path.join(output_dir, f"{model_type}_best_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_type': model_type,
        'hyperparameters': best_params,
        'input_features': input_features,
        'val_metrics': val_metrics,
        'best_epoch': history['best_epoch_indices'][-1],
        'training_history': {
            'best_epochs': history['best_epoch_indices'],
            'best_val_losses': history['best_val_losses']
        }
    }, model_file)
    
    print(f"\nFinal model and results saved to {output_dir}")
    print(f"Best model achieved at epoch {history['best_epoch_indices'][-1]}")
    print(f"Interactive visualizations saved as HTML files for better exploration")
    
    return trained_model