import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum ,rearrange
from cs336_basics.Linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.RoPE import RoPE
class Self_Attention_RoPE(nn.Module):
    def __init__(self,d_in, d_model:int,num_heads:int,max_seq_len, theta, device=None, dtype=None):
        """
            linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert(d_model % num_heads == 0)
        self.d_qkv = d_model // num_heads
        
        self.wq = Linear(d_in,d_model,device, dtype)
        self.wk = Linear(d_in,d_model,device, dtype)
        self.wv = Linear(d_in,d_model,device, dtype)
        self.wo = Linear(d_model,d_model,device, dtype)
        
        self.RoPE = RoPE(theta,self.d_qkv,max_seq_len)
        
    def forward(self, x: torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        batch_size, seq, _ = x.shape
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = rearrange(Q, "... seq (h d) -> ... h seq d",h = self.num_heads,d =self.d_qkv)
        K = rearrange(K, "... seq (h d) -> ... h seq d",h = self.num_heads,d =self.d_qkv)
        V = rearrange(V, "... seq (h d) -> ... h seq d",h = self.num_heads,d =self.d_qkv)
        
        Q = self.RoPE(Q,token_positions)
        K = self.RoPE(K,token_positions)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        attn_output  = scaled_dot_product_attention(Q,K,V,mask)
        attn_output = rearrange(attn_output, '... h s d -> ... s (h d)', h=self.num_heads,d = self.d_qkv)
        output = self.wo(attn_output) 
        return output