"""
Model utility functions.
"""

import torch
import torch.nn as nn


def init_weights(m):
    """
    Initialize model weights using Xavier uniform initialization.
    
    Args:
        m: PyTorch module
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params
