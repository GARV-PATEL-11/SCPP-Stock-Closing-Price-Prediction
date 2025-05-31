# optimizer_factory.py

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler,
)


def get_optimizer(model: nn.Module, params: dict) -> Optimizer:
    """
    Returns the optimizer initialized with model parameters and given hyperparameters.

    Parameters:
    -----------
    model : nn.Module
        The model whose parameters will be optimized.
    
    params : dict
        Dictionary containing:
            - 'optimizer': str, type of optimizer (adam, adamw, radam, sgd)
            - 'learning_rate': float
            - 'weight_decay': float
            - 'momentum': float (only for SGD, optional)

    Returns:
    --------
    optimizer : torch.optim.Optimizer
        Configured optimizer instance.

    Raises:
    -------
    ValueError
        If the optimizer type is unknown.
    """
    optimizer_type = params.get("optimizer", "adam").lower()
    lr = params["learning_rate"]
    weight_decay = params.get("weight_decay", 0.0)

    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "radam":
        # Note: torch.optim does not include RAdam by default
        # You can use torch-optimizer or fallback to Adam
        print("[Warning] RAdam not found in torch.optim. Falling back to Adam.")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "sgd":
        momentum = params.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'")
