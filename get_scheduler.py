# scheduler_factory.py

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler
)
from typing import Union, Optional


def get_scheduler(
    optimizer: Optimizer,
    params: dict
) -> Optional[Union[_LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]]:
    """
    Returns the learning rate scheduler configured with the optimizer and parameters.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer to which the scheduler will be attached.

    params : dict
        Dictionary containing:
            - 'lr_scheduler': str, type of scheduler (none, step, cosine, plateau)
            - scheduler-specific keys (e.g., lr_step_size, lr_gamma, etc.)

    Returns:
    --------
    scheduler : LRScheduler or None
        A PyTorch learning rate scheduler, or None if 'none' is specified.
    """
    scheduler_type = params.get("lr_scheduler", "none").lower()

    if scheduler_type == "none":
        return None

    elif scheduler_type == "step":
        step_size = params.get("lr_step_size", 10)
        gamma = params.get("lr_gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "cosine":
        T_max = params.get("lr_T_max", 10)
        return CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_type == "plateau":
        factor = params.get("lr_factor", 0.5)
        patience = params.get("lr_patience", 5)
        return ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, verbose=True)

    else:
        raise ValueError(f"Unsupported scheduler type: '{scheduler_type}'")
