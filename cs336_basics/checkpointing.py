import torch
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)
    
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location='cpu')  # 可改为需要的设备

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint['iteration']
