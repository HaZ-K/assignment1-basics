import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum
from cs336_basics.Linear import Linear
import math
class SwiGLU(nn.Module):
    def __init__(self, d_model,dff=None, device=None, dtype=None):
        """
            linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        
        super().__init__()
        if dff == None:
            dff = int((8 / 3) * d_model)
            dff = math.ceil(dff / 64) * 64  # 向上取最接近的 64 的倍数
        self.w1 = Linear(d_model, dff,device,dtype)  # 用于 GLU：一半 gate，一半 value
        self.w2 = Linear(dff, d_model,device,dtype)
        self.w3 = Linear(d_model, dff,device,dtype)
       
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        silu = torch.sigmoid(x1) * x1
        x2 = self.w3(x)
        return self.w2(silu * x2)