import torch
import torch.nn as nn
import torch.nn.init as init

class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps =  torch.tensor(eps).to(device=device)
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = self.weight * x / rms
        
        return result.to(in_dtype)
        