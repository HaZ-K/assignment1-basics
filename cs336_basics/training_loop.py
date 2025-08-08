

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable

from tqdm import tqdm

import numpy.typing as npt
import torch
from torch import Tensor
from collections import defaultdict
import heapq
from multiprocessing import Pool
from collections import Counter
from functools import reduce
import regex as re
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.BPETrainer import BPETrainer
from cs336_basics.Linear import Linear
from cs336_basics.Embedding import Embedding
from cs336_basics.RmsNorm import RmsNorm
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.RoPE import RoPE
from cs336_basics.Softmax import softmax
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.Self_Attention import Self_Attention
from cs336_basics.Self_Attention_RoPE import Self_Attention_RoPE
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.AdamW import AdamW 
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpointing import *
import numpy as np
import torch
import numpy as np
import numpy.typing as npt
import wandb
from tqdm import tqdm
from loguru import logger
import torch.nn.functional as F
from cs336_basics.evaluate import evaluate_model



if __name__ == "__main__":
    logger.add("./data/log/train_v0.log", rotation="1 day", retention="7 days", level="INFO")
    wandb_open = True
    # 设置所有超参数
    # 模型参数
    model_config = {
        "vocab_size": 10000,      # 词汇表大小
        "context_length": 256,    # 上下文长度
        "num_layers": 4,          # Transformer Block数
        "num_heads": 16,          # 注意力头数
        "d_model": 512,           # 嵌入空间维度
        "d_ff": 1344,             # 前馈网络维度
        "rope_theta": 10000,      # RoPE参数
    }
    
    # 优化器参数
    optim_config = {
        "lr": 3e-4,               # 学习率
        "weight_decay": 1e-2,     # 权重衰减
        "betas": (0.9, 0.999),    # AdamW的beta参数
        "max_norm": 1.0,          # 梯度裁剪的最大范数
    }
    
    # 训练参数
    train_config = {
        "batch_size": 128,         # 批次大小
        "total_epochs": 10,      # 训练轮数
        "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
        "log_freq": 10,           # 每隔多少步记录一次日志
        "val_freq": 400,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时的批次大小
        "val_batches": 20,        # 验证时使用的批次数量
    }
    
    # 数据路径
    data_paths = {
        "training_dataset_path": "/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/output/TinyStoriesV2-GPT4-valid_by_tiny.bin",
        "validation_dataset_path": "/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/output/TinyStoriesV2-GPT4-valid_by_tiny.bin",  # 验证集路径
        "checkpoint_load_path": None,  # 模型检查点路径
        "checkpoint_save_format": "/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/checkpoint/checkpoint_v0_{}.pt",  # 检查点保存路径格式
        "final_model_path": "/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/checkpoint/final_model_v0.pt",  # 最终模型保存路径
    }
    
    # 初始化wandb
    if wandb_open:
        run = wandb.init(
            project="cs336-assignment-1",
            name="train_v1",
            config={
                "model": model_config,
                "optimizer": optim_config,
                "training": train_config,
            },
            mode="offline"
        )
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    logger.info("开始初始化模型...")
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        
    ).to(device)
    logger.info("模型初始化完成。")

    # 初始化优化器
    logger.info("开始初始化优化器...")
    optimizer = AdamW(
        model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    logger.info("优化器初始化完成。")

    # 如果有checkpoint，则加载checkpoint
    start_iter = 1
    if data_paths["checkpoint_load_path"]:
        logger.info(f"开始加载模型检查点: {data_paths['checkpoint_load_path']}")
        start_iter = load_checkpoint(
            data_paths["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        logger.info(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        logger.info("没有提供模型检查点，开始从头训练。")
    
    # 加载数据集
    logger.info(f"开始加载数据集，训练集：{data_paths['training_dataset_path']}, 验证集：{data_paths['validation_dataset_path']}")
    training_dataset = np.memmap(data_paths['training_dataset_path'], dtype=np.uint16, mode='r')
    
    validation_dataset = None
    if data_paths['validation_dataset_path']:
        validation_dataset = np.memmap(data_paths['validation_dataset_path'], dtype=np.uint16, mode='r')
    logger.info("数据集加载完成")

    # 计算训练所需step
    total_tokens = training_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (train_config["batch_size"] * model_config["context_length"])
    logger.info(f"总token数: {train_config['total_epochs'] * total_tokens}, 训练轮数: {train_config['total_epochs']}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    logger.info(f"总训练步数: {total_steps}")

    # step循环开始
    logger.info("开始训练模型...")
    for step in tqdm(range(start_iter, total_steps + 1), desc="训练进度", unit="step"):
        # 清空梯度
        optimizer.zero_grad()

        # 使用余弦退火更新学习率
        lr_now = learning_rate_schedule(
            step,
            optim_config["lr"],
            optim_config["lr"] * 0.01,
            int(0.05 * total_steps),
            total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        # 获取batch数据
        inputs, targets = get_batch(
            training_dataset,
            batch_size=train_config["batch_size"],
            context_length=model_config["context_length"],
            device=device
        )

        # 前向传播
        
        batch_size,seq_len= inputs.size()
        positions = torch.arange(seq_len)
        token_positions = positions.unsqueeze(0).expand(batch_size, -1)
        logits = model(inputs,token_positions)

        # 计算损失
        
        loss = cross_entropy(logits, targets)

        # 反向传播和优化参数
        loss.backward()
        
        gradient_clipping(model.parameters(), max_norm=optim_config["max_norm"]) # 梯度裁剪
        optimizer.step()

        # 日志记录
        if step % train_config["log_freq"] == 0:
            logger.info(f"Step {step}, Loss: {loss.item()}")

            # 使用wandb记录损失和梯度范数
            if wandb_open:
                wandb.log({"train_loss": loss.item(), "lr": lr_now, "step": step})
        
        # 在验证集上评估模型
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            logger.info(f"在验证集上评估模型...")
            val_loss = evaluate_model(
                model=model,
                dataset=validation_dataset,
                device=device,
                batch_size=train_config["val_batch_size"],
                context_length=model_config["context_length"],
                num_batches=train_config["val_batches"]
            )
            logger.info(f"验证集损失: {val_loss}")
            if wandb_open:
                wandb.log({"val_loss": val_loss, "step": step})
        
        # 保存检查点
        if step % train_config["checkpoint_freq"] == 0:
            checkpoint_save_path = data_paths["checkpoint_save_format"].format(step)
            logger.info(f"正在保存模型检查点到: {checkpoint_save_path}")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_save_path
            )
            logger.info("模型检查点保存成功。")
    logger.info("模型训练完成。")
    
    # 保存最终模型
    # logger.info(f"正在保存最终模型到: {data_paths["final_model_path"]}")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=data_paths["final_model_path"],
    )
    logger.info("最终模型保存成功。")
    
    # 关闭wandb
    # wandb.finish()