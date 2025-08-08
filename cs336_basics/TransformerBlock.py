import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum ,rearrange
from cs336_basics.Linear import Linear
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.Self_Attention_RoPE import Self_Attention_RoPE
from cs336_basics.Self_Attention import Self_Attention
from cs336_basics.RmsNorm import RmsNorm

class TransformerBlock(nn.Module):
    def __init__(self,d_in, d_model:int,num_heads:int, dff:int,max_seq_len,theta, device=None, dtype=None):
       super().__init__()
       self.attention = Self_Attention_RoPE(d_in,d_model,num_heads,max_seq_len,theta)
       self.norm1 = RmsNorm(d_model)
       self.norm2 = RmsNorm(d_model)
       self.ffn = SwiGLU(d_model,dff)
      
    def forward(self, x: torch.Tensor,token_positions) -> torch.Tensor:
        x1 = x + self.attention(self.norm1(x),token_positions)
        y1 = x1 + self.ffn(self.norm2(x1))
        return y1