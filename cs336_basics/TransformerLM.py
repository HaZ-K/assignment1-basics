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
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.Softmax import softmax
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta):
       super().__init__()
       self.vocab_size = vocab_size
       self.context_length = context_length
       self.num_layers = num_layers
       self.d_model = d_model
       self.heads = num_heads
       self.d_ff = d_ff
       self.theta = rope_theta
       self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
    
       self.embedding = Embedding(vocab_size,d_model)  
       self.norm = RmsNorm(d_model)
       self.linear = Linear(d_model,vocab_size)
    
    def forward(self, x: torch.Tensor,token_positions) -> torch.Tensor:
        res = self.embedding(x)
        
        for block in self.blocks:
            res = block(res, token_positions)
        return self.linear(self.norm(res))