import torch
from torch.optim import Optimizer
import math

def learning_rate_schedule(t, alpha_max, alpha_min, tw ,tc):
    if t < tw:
        # Warm-up 阶段：线性上升
        return t / tw * alpha_max
    elif tw <= t <= tc:
        # Cosine annealing 阶段
        cos_inner = math.pi * (t - tw) / (tc - tw)
        cos_out = math.cos(cos_inner)
        return alpha_min + 0.5 * (1 + cos_out) * (alpha_max - alpha_min)
    else:
        # Post-annealing 阶段：保持最小学习率
        return alpha_min
    