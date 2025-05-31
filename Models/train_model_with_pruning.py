import torch  # PyTorch library for defining and training neural networks
import optuna  # Optuna for hyperparameter optimization and pruning during training
import numpy as np  # NumPy for numerical computations during evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
from Utils.get_loss_function import get_loss_function
from Utils.get_scheduler import get_scheduler
from Utils.get_optimizer import get_optimizer

def train_model_with_pruning(model, train_loader, val_loader, num_epochs=100, patience=20, 
                             trial=None, pruning_callback=None):
    """
    Train model with pruning capability using Optuna for early stopping based on validation loss.
    
    This function trains the model, calculates loss during training and validation, and
    integrates pruning with Optuna for early stopping based on the validation loss.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if trial is not None:
        params = trial.params
    else:
        params = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'lr_scheduler': 'none',
            'loss_function': 'mse'
        }
    
    criterion = get_loss_function(params)
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(optimizer, params)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history