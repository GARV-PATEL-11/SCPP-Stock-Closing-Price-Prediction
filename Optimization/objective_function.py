# objective_function.py

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from optuna.integration import PyTorchLightningPruningCallback
from Models.train_model_with_pruning import train_model_with_pruning
from Models.create_model_with_params import create_model_with_params
from Optimization.suggest_hyperparameters import suggest_hyperparameters
from Models.evaluate_model import evaluate_model

def objective_function(trial, model_type, train_loader, val_loader, num_features, output_dir):
    """
    Objective function for Optuna to perform multi-fidelity hyperparameter optimization.

    This function adapts the amount of computation (epochs, patience, training subset) based on a fidelity ratio.
    The fidelity gradually increases across trials, allowing faster exploration in early trials and deeper training
    in later ones. It returns the validation RMSE to be minimized.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        An Optuna trial object used to suggest hyperparameters and track performance.
    model_type : str
        Type of model to optimize ('lstm', 'gru', etc.).
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    num_features : int
        Number of input features for the model.
    output_dir : str
        Directory path to save results or model artifacts.

    Returns:
    --------
    float
        Validation RMSE (Root Mean Squared Error), which is the objective to minimize.
    """
    
    # Step 1: Compute fidelity ratio (progressive depth based on trials completed)
    fidelity_ratio = min(1.0, trial.number / (len(trial.study.trials) * 0.7))

    # Step 2: Suggest hyperparameters
    params = suggest_hyperparameters(trial, model_type)

    # Step 3: Set scaled training config
    epochs_full = 250
    patience_full = 50
    epochs = max(20, int(epochs_full * fidelity_ratio))
    patience = max(10, int(patience_full * fidelity_ratio))

    # Step 4: Initialize model
    model = create_model_with_params(model_type, num_features, params)
    model.float()

    # Step 5: Use subset of data if low fidelity
    if fidelity_ratio < 0.5:
        subset_ratio = 0.3 + fidelity_ratio
        subset_size = int(len(train_loader.dataset) * subset_ratio)
        indices = torch.randperm(len(train_loader.dataset))[:subset_size]
        subset = torch.utils.data.Subset(train_loader.dataset, indices)
        current_train_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True)
    else:
        current_train_loader = train_loader

    # Step 6: Set up pruning callback
    pruning_callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

    # Step 7: Train model with early stopping and pruning
    start_time = time.time()
    trained_model, history = train_model_with_pruning(
        model=model,
        train_loader=current_train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        patience=patience,
        trial=trial,
        pruning_callback=pruning_callback
    )
    training_time = time.time() - start_time

    # Step 8: Evaluate model on validation data
    val_results = evaluate_model(trained_model, val_loader)
    validation_rmse = val_results['rmse']

    # Step 9: Log additional metrics to Optuna trial
    trial.set_user_attr("val_mae", val_results['mae'])
    trial.set_user_attr("val_r2", val_results['r2'])
    trial.set_user_attr("val_mape", val_results['mape'])
    trial.set_user_attr("training_time", training_time)
    trial.set_user_attr("epochs_completed", len(history['train_loss']))
    trial.set_user_attr("fidelity_ratio", fidelity_ratio)

    # Step 10: Handle invalid metrics
    if np.isnan(validation_rmse):
        return float('inf')

    # Step 11: Return the objective to minimize
    return validation_rmse

