from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.Softmax import softmax
from cs336_basics.checkpointing import *
from cs336_basics.AdamW import AdamW
import torch
def decoding(model:TransformerLM,tokenizer:BPETokenizer,prompt:str,max_token=4096,temperature=0.1,top_p=10,device: str = "cuda" if torch.cuda.is_available() else "cpu")->str:
    model.eval()
    model.to(device)
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)
    end_token_id = tokenizer.encode("<|endoftext|>")[0]
    print(f"end_token_id : {end_token_id}")
    for _ in range(max_token):
        if input_ids.shape[1] > model.context_length:
            input_ids = input_ids[:, -model.context_length:]
        with torch.no_grad():
            batch_size,seq_len= input_ids.size()
            positions = torch.arange(seq_len)
            token_positions = positions.unsqueeze(0).expand(batch_size, -1)
            
            logits = model(input_ids,token_positions)  # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :]
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        probs = softmax(next_token_logits, d_model= model.d_model)
        # print(f"probs : {probs}")
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        if torch.any(cutoff):
            cutoff_idx = torch.where(cutoff)[0][0]
            sorted_probs = sorted_probs[:cutoff_idx + 1]
            sorted_indices = sorted_indices[:cutoff_idx + 1]
        sorted_probs = sorted_probs / sorted_probs.sum()
        # print(f"sorted : {sorted_probs}")
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices[sampled_index]
        # 拼接生成
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == end_token_id:
            break
    output_ids = input_ids[0].tolist()
    decoded_text = tokenizer.decode(output_ids,end_token_id)
    return decoded_text

if __name__ == "__main__":
    model_config = {
        "vocab_size": 10000,      # 词汇表大小
        "context_length": 256,    # 上下文长度
        "num_layers": 4,          # Transformer Block数
        "num_heads": 16,          # 注意力头数
        "d_model": 512,           # 嵌入空间维度
        "d_ff": 1344,             # 前馈网络维度
        "rope_theta": 10000,      # RoPE参数
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
    ).to(device)
    model = torch.compile(model)
    optim_config = {
        "lr": 3e-4,               # 学习率
        "weight_decay": 1e-2,     # 权重衰减
        "betas": (0.9, 0.999),    # AdamW的beta参数
        "max_norm": 1.0,          # 梯度裁剪的最大范数
    }
    optimizer = AdamW(
        model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    
    vocab_filepath = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/TinyStories_train_10000_token_vocab.bin'
    mergers_filepath = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/TinyStories_train_10000_merges.bin'
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab_filepath,mergers_filepath,special_tokens)
    src= '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/checkpoint/final_model_v0.pt'
    load_checkpoint(src,model=model,optimizer=optimizer)
    
    res = decoding(model,tokenizer,prompt="Once upon a time, in a",max_token=256,temperature=0.5,top_p=0.7)
    
    print(res)
    