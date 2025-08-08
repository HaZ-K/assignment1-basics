
import numpy.typing as npt
import numpy as np
import torch
import os
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    # 保证采样起点不越界
    max_start = n - context_length - 1
    if max_start <= 0:
        raise ValueError("Dataset 太短，无法生成有效样本。")

    # 随机采样 batch_size 个起点位置
    starts = np.random.randint(0, max_start, size=batch_size)

    # 构造输入和目标序列
    x = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy((dataset[i+1 : i + context_length+1]).astype(np.int64)) for i in starts])
    x,y = x.to(device),y.to(device)
    # 转为 PyTorch tensor 并放到目标设备上
    return x,y

def get_batch_mmap(dataset: str, batch_size: int, context_length: int, device: str):
    
    data = np.memmap(os.path.join(dataset), dtype=np.uint16, mode='r')
    
    starts = torch.randint(len(data) - context_length, (batch_size,))

    # 构造输入和目标序列
    x = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy((dataset[i+1 : i + context_length+1]).astype(np.int64)) for i in starts])
    x,y = x.to(device),y.to(device)
    # 转为 PyTorch tensor 并放到目标设备上
    
    return x,y