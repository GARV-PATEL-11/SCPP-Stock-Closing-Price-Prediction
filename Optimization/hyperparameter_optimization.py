# hyperparameter_optimization.py

import os
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial
from datetime import datetime
from Optimization.prepare_data_for_hpo import prepare_data_for_hpo
from Optimization.objective_function import objective_function
from Models.train_final_model import train_final_model
from Optimization.save_optimization_results import save_optimization_results

# External dependencies assumed to be implemented elsewhere

def run_hyperparameter_optimization(Modelling_data, output_dir='hpo_results', n_trials=200, model_type='lstm'):
    """
    Run Bayesian Multi-Fidelity hyperparameter optimization for the specified model type using Optuna.

    Parameters:
    -----------
    Modelling_data : str
        Path to the input data file or identifier for dataset loading.
    output_dir : str
        Directory to save optimization results. Default is 'hpo_results'.
    n_trials : int
        Number of optimization trials to perform. Default is 200.
    model_type : str
        Type of model to optimize ('lstm', 'gru', etc.). Default is 'lstm'.

    Returns:
    --------
    best_params : dict
        Dictionary of the best hyperparameters found during optimization.
    study : optuna.Study
        The Optuna study object containing the optimization history.
    """
    print(f"Starting Bayesian Multi-Fidelity HPO with {n_trials} trials for {model_type} model")

    # Step 1: Create results directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Prepare data
    data, train_loader, val_loader, scaler_X, scaler_y, input_features, num_features = prepare_data_for_hpo(
        Modelling_data, model_type
    )

    # Step 3: Set up Optuna pruner
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=15,
        interval_steps=10
    )

    # Step 4: Set up Optuna sampler
    sampler = TPESampler(
        seed=42,
        n_startup_trials=25,
        multivariate=True,
        group=True
    )

    # Step 5: Create Optuna study
    study_name = f"{model_type}_multifidelity_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=False
    )

    # Step 6: Define objective function with additional args using partial
    objective = partial(
        objective_function,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=num_features,
        output_dir=output_dir
    )

    # Step 7: Start optimization
    try:
        study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # Step 8: Retrieve best results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\nHyperparameter Optimization Complete!")
    print(f"Best {model_type} params: {best_params}")
    print(f"Best validation RMSE: {best_value:.8f}")

    # Step 9: Save study and results
    save_optimization_results(study, model_type, output_dir)

    # Step 10: Train final model with best hyperparameters
    train_final_model(
        best_params,
        model_type,
        train_loader,
        val_loader,
        num_features,
        scaler_y,
        input_features,
        data,
        output_dir
    )

    return best_params, study
