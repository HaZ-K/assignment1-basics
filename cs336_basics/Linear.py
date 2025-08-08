import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
            linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features,device=device, dtype=dtype))
        init.trunc_normal_(self.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return einsum(x,self.weight,"... d_in, d_out d_in -> ... d_out")