import os  # OS module for file and directory handling
import pandas as pd  # Pandas for creating and saving DataFrames
from datetime import datetime  # For generating timestamps
import optuna  # Optuna for hyperparameter optimization and study handling
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)  # Optuna visualization tools


def save_optimization_results(study, model_type, output_dir):
    """
    Save optimization results from the Optuna study to files.
    
    This function saves the results of hyperparameter optimization, including study statistics,
    trial data, and visualizations, to specified files in the output directory.

    Args:
        study (optuna.Study): The Optuna study object containing the optimization results.
        model_type (str): The type of model (e.g., 'lstm', 'gru', etc.).
        output_dir (str): The directory to save the results.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study statistics to a text file
    stats_file = os.path.join(output_dir, f"{model_type}_hpo_stats_{timestamp}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Study name: {study.study_name}\n")
        f.write(f"Number of completed trials: {len(study.trials)}\n")
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best value (RMSE): {study.best_value:.8f}\n\n")
        
        f.write("Best hyperparameters:\n")
        for param_name, param_value in study.best_params.items():
            f.write(f"  {param_name}: {param_value}\n")
        
        f.write("\nBest trial user attributes:\n")
        for key, value in study.best_trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")
    
    # Save trial data to a CSV file
    trials_file = os.path.join(output_dir, f"{model_type}_hpo_trials_{timestamp}.csv")
    trials_data = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_dict = {
                'number': trial.number,
                'value': trial.value,
                **trial.params,
                **trial.user_attrs
            }
            trials_data.append(trial_dict)
    
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv(trials_file, index=False)
    
    # Plot optimization history
    try:
        fig_history = plot_optimization_history(study)
        history_file = os.path.join(output_dir, f"{model_type}_history_plot_{timestamp}.png")
        fig_history.savefig(history_file)
    except Exception as e:
        print(f"Could not create optimization history plot: {e}")
    
    # Plot parameter importances
    try:
        fig_importance = plot_param_importances(study)
        importance_file = os.path.join(output_dir, f"{model_type}_importance_plot_{timestamp}.png")
        fig_importance.savefig(importance_file)
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")
    
    # Plot parallel coordinates
    try:
        fig_parallel = plot_parallel_coordinate(study)
        parallel_file = os.path.join(output_dir, f"{model_type}_parallel_plot_{timestamp}.png")
        fig_parallel.savefig(parallel_file)
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")
    
    print(f"Optimization results saved to {output_dir}")
