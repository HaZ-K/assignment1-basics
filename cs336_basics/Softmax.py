import torch
import torch.nn as nn
import math

def softmax(x:torch.tensor,d_model:int) -> torch.tensor:
    # Subtract the maximum value for numerical stability
    # This prevents exp() from overflowing
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Compute softmax
    exp_x = torch.exp(x_shifted)
    softmax_output = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    return softmax_output