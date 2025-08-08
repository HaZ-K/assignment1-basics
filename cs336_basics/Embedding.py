import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
            linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        init.trunc_normal_(self.weight, std=0.02)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.weight[token_ids]