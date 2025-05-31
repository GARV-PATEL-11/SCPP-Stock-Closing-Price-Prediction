import os
import torch
import numpy as np
from datetime import datetime

def train_model_with_best_checkpoints(model, train_loader, val_loader, model_type, best_params, output_dir="Results", num_epochs=1000, patience=500):
    if best_params is None:
        raise ValueError("best_params cannot be None. Please ensure a valid dictionary is provided.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if hasattr(model, 'apply_config'):
        model.apply_config(best_params)

    checkpoint_dir = os.path.join(output_dir, "model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer_type = best_params['optimizer'].lower()
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    momentum = best_params.get('momentum', 0.9) 

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'radam':
        print("[Warning] RAdam not found in torch.optim. Falling back to Adam.")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = momentum 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'")

    scheduler_type = best_params['lr_scheduler'].lower()
    if scheduler_type == 'step':
        step_size = best_params['lr_step_size']
        gamma = best_params['lr_gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = best_params['lr_T_max']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'plateau':
        factor = best_params['lr_factor']
        plateau_patience = best_params['lr_patience']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=plateau_patience, verbose=True
    )
    elif scheduler_type == 'none':
        scheduler = None 
    else:
        raise ValueError(f"Unsupported scheduler type: '{scheduler_type}'")

    loss_fn_name = best_params['loss_function'].lower()
    if loss_fn_name == 'huber':
        delta = best_params['delta']
        loss_fn = torch.nn.HuberLoss(delta=delta)
    elif loss_fn_name == 'mae':
        loss_fn = torch.nn.L1Loss()
    else:
        loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'best_epoch_indices': [], 'best_val_losses': [], 'epochs': []}
    best_model_paths = []
    best_model_metrics = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def evaluate_validation():
        if val_loader is None:
            return avg_train_loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss = loss_fn(y_pred, y_batch)
                val_losses.append(val_loss.item())
        return np.mean(val_losses)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = evaluate_validation()

        if scheduler is not None:            if scheduler_type == 'plateau':        
                scheduler.step(avg_val_loss)  # requires validation los        s
            else:
                scheduler.step()  # for step, cosine, etc.


        current_lr = optimizer.param_groups[0]['lr']
        history['epochs'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            model_path = os.path.join(checkpoint_dir, f"{model_type}_epoch_{epoch}_{timestamp}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'lr': current_lr
            }, model_path)

            history['best_epoch_indices'].append(epoch)
            history['best_val_losses'].append(avg_val_loss)
            best_model_paths.append(model_path)
            best_model_metrics.append({
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'path': model_path
            })
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f} (Best Model Saved)")
        else:
            counter += 1
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    best_model_idx = history['best_val_losses'].index(min(history['best_val_losses']))
    best_model_path = best_model_paths[best_model_idx]
    print(f"Loading overall best model from epoch {history['best_epoch_indices'][best_model_idx]}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for path in best_model_paths:
        if path != best_model_path:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")

    return model, history, best_model_metrics
