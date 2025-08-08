import os
import numpy as np
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm
import regex as re

class BPETokenizer:
    def __init__(self, vocab:Dict[int, bytes], merges:List[Tuple[bytes, bytes]], special_tokens:List[str] = None) -> None:
        """
        初始化BPE分词器
        """
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None
        self.special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r"(?!)"
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
      
        self.token_vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id: Dict[bytes,int] = {token: idx for idx, token in self.token_vocab.items()}
        # self.pre_tokenizer = PreTokenizer(self.special_tokens)

        self.word_to_ids: Dict[bytes, List[int]] = {} # 缓存已经计算过的词的对应id序列
    
    @classmethod
    def from_files(cls, vocab_filepath: str, mergers_filepath: str, special_tokens: List[str] = None) -> 'BPETokenizer':
        """
        从文件加载BPE分词器
        
        Args:
            vocab_filepath: 词汇表文件路径
            mergers_filepath: 合并规则文件路径
            special_tokens: 特殊token列表
        
        Returns:
            BPETokenizer 实例
        """
        # 读取vocab
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            # 读取词汇表大小
            vocab_size_bytes = f.read(4)
            vocab_size = int.from_bytes(vocab_size_bytes, byteorder='little')
            
            # 读取每个token: <id(4字节)><长度(4字节)><token内容(bytes)>
            for _ in range(vocab_size):
                token_id_bytes = f.read(4)
                token_id = int.from_bytes(token_id_bytes, byteorder='little')
                
                token_len_bytes = f.read(4)
                token_len = int.from_bytes(token_len_bytes, byteorder='little')
                
                token = f.read(token_len)
                vocab[token_id] = token
        
        # 读取merges
        merges = []
        with open(mergers_filepath, 'rb') as f:
            # 读取合并规则数量
            merges_count_bytes = f.read(4)
            merges_count = int.from_bytes(merges_count_bytes, byteorder='little')
            
            # 读取每个合并规则: <第一部分长度(4字节)><第一部分内容(bytes)><第二部分长度(4字节)><第二部分内容(bytes)>
            for _ in range(merges_count):
                first_len_bytes = f.read(4)
                first_len = int.from_bytes(first_len_bytes, byteorder='little')
                
                first = f.read(first_len)
                
                second_len_bytes = f.read(4)
                second_len = int.from_bytes(second_len_bytes, byteorder='little')
                
                second = f.read(second_len)
                
                merges.append((first, second))
        
        return cls(vocab, merges, special_tokens)

    def calculate_token_ids(self, word: bytes) -> List[int]:
        """
        将一个bytes根据merges不断合并，得到其token ID序列
        """
        token_ids = []
        # 将每个字节作为独立的bytes对象
        bytes_list = [bytes([b]) for b in word]  # 将每个字节作为单独的bytes对象

        while len(bytes_list) > 1:
            # 一轮中可能同时满足多个合并规则，选择index最小的合并规则进行合并
            min_rule_idx = None
            min_merge_pos = None
            
            # 遍历当前字节列表中所有可能的合并规则
            for i, pair in enumerate(zip(bytes_list[:-1], bytes_list[1:])):
                idx = self.token_to_id.get(pair[0] + pair[1])
                if (idx is not None) and ((min_rule_idx is None) or (idx < min_rule_idx)):
                    # 找到一个更小的合并规则，更新最小index和位置
                    min_rule_idx = idx
                    min_merge_pos = i
            
            if min_rule_idx is None:
                # 没有可合并的规则
                break
            
            # 执行合并
            bytes_list[min_merge_pos:min_merge_pos + 2] = [bytes_list[min_merge_pos] + bytes_list[min_merge_pos + 1]]
        
        # 出循环说明已经合并完成，开始翻译为ids
        for part in bytes_list:
            try:
                id = self.token_to_id[part]
                token_ids.append(id)
            except KeyError:
                # 如果没有找到对应的ID，可能是未训练的token,暂时不处理
                print(f"Warning: Token {part} not found in vocabulary.")
                pass
        return token_ids
    
    def encode(self, text:str) -> List[int]:
        """
        将文本编码为BPE token ID列表
        """
        # 预分词，把str转为list[bytes]
        words = self.pretokenize(text) # word是bytes
        ids = []
        for word in words:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
                ids.append(self.token_to_id[word])
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                ids.extend(self.word_to_ids[word])
            else:
                # 计算该词对应token ID序列
                token_ids = self.calculate_token_ids(word)
                self.word_to_ids[word] = token_ids  # 缓存结果
                ids.extend(token_ids)
        return ids
    
    def pretokenize(self, text: str) -> List[bytes]:
        """
        将输入文本预分词，包括常规token和特殊token
        
        Args:
            text: 输入文本
        
        Returns:
            预分词结果bytes列表
        """
        # 首先按特殊token进行分割，但保留特殊token
        parts = re.split(f'({self.special_tokens_pattern})', text)
        
        result = []
        
        for part in parts:
            if part in self.special_tokens:
                # 特殊token直接放入
                result.append(part.encode('utf-8'))
            elif part:  # 跳过空字符串
                # 对普通文本使用word_pattern进行分词
                tokens = [match.group(0).encode('utf-8') for match in self.word_pattern.finditer(part)]
                result.extend(tokens)
        
        return result
    def pretokenize_iter(self,texts: Iterable[str]):
        for text in texts:
            # 首先按特殊token进行分割，但保留特殊token
            parts = re.split(f'({self.special_tokens_pattern})', text)
            for part in parts:
                if part in self.special_tokens:
                    # 特殊token直接生成
                    yield part.encode('utf-8')
                elif part:  # 跳过空字符串
                    # 对普通文本使用word_pattern进行分词并生成
                    for match in self.word_pattern.finditer(part):
                        yield match.group(0).encode('utf-8')
                        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        对可迭代对象（例如文件句柄）中的每个文本进行编码，每次调用返回一个token ID
        """
        words_iter = self.pretokenize_iter(iterable)
        for word in words_iter:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
               
                yield self.token_to_id[word]
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                
                yield from self.word_to_ids[word]
            else:
                # 计算该词对应token ID序列
                token_ids = self.calculate_token_ids(word)
                
                self.word_to_ids[word] = token_ids
                
                yield from token_ids

    def encode_to_npfile(self, input_path: os.PathLike, output_path: os.PathLike) -> None:
        """
        将输入文件中的文本编码为BPE token ID列表，并保存为numpy数组文件.npy
        """
        import tempfile

        # # 计算文件总行数用于进度条
        # total_lines = 0
        # with open(input_path, 'r', encoding='utf-8') as f:
        #     for _ in f:
        #         total_lines += 1
        
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            token_count = 0
            
            with open(input_path, 'r', encoding='utf-8') as f:
                # 使用tqdm创建进度条
                for token_id in tqdm(self.encode_iterable(f), desc="编码进度", unit="token", total=None):
                    tmpfile.write(np.uint16(token_id).tobytes())
                    
                    token_count += 1

            tmpfile_path = tmpfile.name

        # 读取临时文件创建memmap，并保存为.npy
        print("正在保存到文件...")
        mm_array = np.memmap(tmpfile_path, dtype=np.uint16, mode='r', shape=(token_count,))
        print(mm_array)
        mm_array.tofile(output_path)
        # np.save(output_path, mm_array)
        # del mm_array
        # os.remove(tmpfile_path)
        # data = np.load(output_path)
        data = np.memmap(output_path, dtype=np.uint16, mode='r')
        print(data)
        file_bytes = os.path.getsize(input_path)
        compression_ratio = file_bytes / token_count
        print(f"文件已保存至：{output_path}\n总token数：{token_count},压缩率（Compression ratio）：{compression_ratio:.2f} bytes/token")

    def decode(self, ids: Iterable[int], end_token_id: int = None) -> str:
        """
        将BPE token ID列表解码为文本
        """
        text_bytes = b""
        for id in ids:
            if id in self.token_vocab:
                text_bytes += self.token_vocab[id]
            else:
                print(f"Warning: ID {id} not found in vocabulary.")
                continue

            if (end_token_id is not None) and (id == end_token_id):
                break
            
        return text_bytes.decode('utf-8', errors='ignore')

if __name__ == "__main__":
    
    # vocab_filepath = './workshop/owt_train_32000vocab.bin'
    # mergers_filepath = './workshop/owt_train_32000merges.bin'
    vocab_filepath = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/TinyStories_train_10000_token_vocab.bin'
    mergers_filepath = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/TinyStories_train_10000_merges.bin'
    special_tokens = ["<|endoftext|>"]
   
    tokenizer = BPETokenizer.from_files(vocab_filepath,mergers_filepath,special_tokens)
    input_path = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    output_path = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/workshop/output/TinyStoriesV2-GPT4-train_by_tiny.bin'
    tokenizer.encode_to_npfile(input_path,output_path)
    data = np.memmap(output_path, dtype=np.uint16, mode='r')
    print(data)
    data = data.tolist()
    # print()
    data = tokenizer.decode(data[:10000])
    print(data)
   
