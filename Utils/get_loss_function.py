# loss_factory.py

import torch
import torch.nn as nn


def get_loss_function(params: dict) -> nn.Module:
    """
    Return an appropriate PyTorch loss function based on the provided parameters.

    Parameters:
    -----------
    params : dict
        Dictionary of configuration parameters. Must include:
            - 'loss_function': str
                Type of loss function to use. Supported values: 'mse', 'mae', 'huber'.
            - 'delta': float (optional)
                The delta value for the Huber loss (used only if loss_function is 'huber').

    Returns:
    --------
    loss_function : nn.Module
        An instantiated PyTorch loss function.

    Raises:
    -------
    ValueError
        If the provided loss function type is unsupported.
    """
    loss_type = params.get("loss_function", "mse").lower()

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type == "huber":
        delta = params.get("delta", 1.0)
        return nn.HuberLoss(delta=delta)

    else:
        raise ValueError(f"Unsupported loss function type: '{loss_type}'")
