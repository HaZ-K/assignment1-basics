import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        position = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(position, inv_freq)  # (max_seq_len, d_k//2)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]

        x_reshaped = x.view(*batch_shape, seq_len, self.d_k // 2, 2)

        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        
        #poor code!
        cos = cos.unsqueeze(-1).unsqueeze(1)
        sin = sin.unsqueeze(-1).unsqueeze(1)
        
        # print(x_reshaped[..., 0].size())
        # print( cos.squeeze(-1).size())
        x_rotated = torch.stack(
            [
                x_reshaped[..., 0] * cos.squeeze(-1) - x_reshaped[..., 1] * sin.squeeze(-1),
                x_reshaped[..., 1] * cos.squeeze(-1) + x_reshaped[..., 0] * sin.squeeze(-1),
            ],
            dim=-1
        )

        return x_rotated.view(*batch_shape, seq_len, self.d_k)