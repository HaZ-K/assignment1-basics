# adapters.py

import torch

def gradient_clipping(params, max_norm, eps=1e-6):
    """
    对所有参数的梯度进行 L2 范数裁剪，最大值为 max_norm。
    参数:
        params: 可迭代的模型参数列表（例如 model.parameters()）
        max_norm: 允许的最大 L2 范数
        eps: 数值稳定性使用的小常数（默认 1e-6）
    """
    # 获取所有存在 grad 的参数，并将它们的梯度拼接为一个向量
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return  # 没有梯度，无需裁剪

    # 拼接所有梯度为一个向量
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), p=2) for g in grads]),
        p=2
    )

    # 计算缩放因子
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)  # 原地缩放梯度
