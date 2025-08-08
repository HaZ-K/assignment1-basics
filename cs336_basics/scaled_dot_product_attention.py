import torch
import torch.nn as nn
import math
from cs336_basics.Softmax import softmax
from einops import einsum
def scaled_dot_product_attention(q:torch.tensor,k:torch.tensor,v:torch.tensor,mask=None) -> torch.tensor:
    attn_scores = einsum(q,k,"... queries d_k,... keys d_k ->... queries keys")
    d_k = k.size(-1)
    attn_scores = attn_scores / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    attn_scores = softmax(attn_scores,d_k)
    
    output = einsum(attn_scores, v,'... queries keys, ... keys d_v-> ... queries d_v', )
    return output
   