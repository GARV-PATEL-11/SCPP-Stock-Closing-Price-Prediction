# ======================== Imports ===========================
import os  # For file and directory handling
import pandas as pd  # For saving trial data as CSV
import numpy as np  # For numerical sorting
import seaborn as sns  # For custom visualizations
import matplotlib.pyplot as plt  # For plotting figures
from datetime import datetime  # For timestamping outputs
import optuna  # For hyperparameter optimization
from optuna.visualization import plot_parallel_coordinate as optuna_plot_parallel_coordinate  # Native Optuna plot

# ======================== Custom Plot Functions ===========================

def plot_parallel_coordinate(study):
    """Create parallel coordinate plot using Optuna's built-in visualization."""
    return optuna_plot_parallel_coordinate(study).figure

def plot_param_importances(study):
    """Create parameter importance plot using Seaborn."""
    param_importances = optuna.importance.get_param_importances(study)
    
    params = list(param_importances.keys())
    importances = list(param_importances.values())
    indices = np.argsort(importances)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=[importances[i] for i in indices],
        y=[params[i] for i in indices],
        palette="viridis"
    )
    
    plt.xlabel("Importance")
    plt.ylabel("Parameter")
    plt.title("Parameter Importances")
    plt.tight_layout()
    return plt.gcf()

def plot_optimization_history(study):
    """Create optimization history plot using Seaborn."""
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_numbers = [t.number for t in complete_trials]
    trial_values = [t.value for t in complete_trials]
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=trial_numbers, y=trial_values, marker='o', alpha=0.7)
    
    best_trial = study.best_trial
    plt.scatter(best_trial.number, best_trial.value, color='red', s=100, marker='*',
                label=f"Best Trial (RMSE={best_trial.value:.4f})")
    
    # Handle pruned trials (optional, gray crosses)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned_trials:
        pruned_numbers = [t.number for t in pruned_trials]
        pruned_values = [
            t.intermediate_values[max(t.intermediate_values.keys())]
            if t.intermediate_values else float('nan')
            for t in pruned_trials
        ]
        sns.scatterplot(x=pruned_numbers, y=pruned_values, color='gray',
                        marker='x', alpha=0.5, label="Pruned Trials")
    
    plt.title("Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

# ======================== Save Function ===========================

def save_optimization_results(study, model_type, output_dir):
    """
    Save Optuna hyperparameter optimization results including statistics,
    trial data, and visualizations.
    
    Args:
        study (optuna.Study): The Optuna study object.
        model_type (str): Model identifier (e.g., 'lstm', 'gru').
        output_dir (str): Directory to save the output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ---- Save study statistics ----
    stats_file = os.path.join(output_dir, f"{model_type}_hpo_stats_{timestamp}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Study name: {study.study_name}\n")
        f.write(f"Number of completed trials: {len(study.trials)}\n")
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best value (RMSE): {study.best_value:.8f}\n\n")
        f.write("Best hyperparameters:\n")
        for param, val in study.best_params.items():
            f.write(f"  {param}: {val}\n")
        f.write("\nBest trial user attributes:\n")
        for k, v in study.best_trial.user_attrs.items():
            f.write(f"  {k}: {v}\n")
    
    # ---- Save all completed trials as CSV ----
    trials_file = os.path.join(output_dir, f"{model_type}_hpo_trials_{timestamp}.csv")
    trials_data = [
        {
            'number': trial.number,
            'value': trial.value,
            **trial.params,
            **trial.user_attrs
        }
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    pd.DataFrame(trials_data).to_csv(trials_file, index=False)
    
    # ---- Save visualizations ----
    try:
        fig = plot_optimization_history(study)
        fig.savefig(os.path.join(output_dir, f"{model_type}_history_plot_{timestamp}.png"))
    except Exception as e:
        print(f"[Warning] Failed to save optimization history plot: {e}")
    
    try:
        fig = plot_param_importances(study)
        fig.savefig(os.path.join(output_dir, f"{model_type}_importance_plot_{timestamp}.png"))
    except Exception as e:
        print(f"[Warning] Failed to save parameter importance plot: {e}")
    
    try:
        fig = plot_parallel_coordinate(study)
        fig.savefig(os.path.join(output_dir, f"{model_type}_parallel_plot_{timestamp}.png"))
    except Exception as e:
        print(f"[Warning] Failed to save parallel coordinate plot: {e}")
    
    print(f"[Info] Optimization results saved in: {output_dir}")
