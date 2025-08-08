from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from tqdm import tqdm
import multiprocessing
import numpy.typing as npt
import torch
from torch import Tensor
from collections import defaultdict
import heapq
from multiprocessing import Pool
from collections import Counter
from functools import reduce
import regex as re
from torch.profiler import profile, record_function, ProfilerActivity
import io

def build_word_frequency(docs: Iterable[str],) -> Counter:
        """构建词频字典"""
        word_freq = Counter()
        str_freq = Counter()
        word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for doc in docs:
        
            if not doc:
                # 跳过空字符串
                continue
            # 收集doc中所有单词
            matches = [word.group(0) for word in word_pattern.finditer(doc)]
            # 批量更新计数器
            str_freq.update(matches)
        # 将字符串词频转换为字节词频
        for word, freq in str_freq.items():
            word_freq[word.encode('utf-8')] = freq
        return word_freq
class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str],vocab=None, merges=None):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 词汇表大小
            special_tokens: 特殊token列表
        """
        self.num_processes = 36
        self.vocab_size = vocab_size
       
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else r"(?!)"
        self.word_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.token_vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.splits: dict[bytes, list[bytes]] = {}  # b"going" -> [b'g', b'o', b'ing'] 可以以此知道当前word有哪些pair
        self.pair_freqs: dict[tuple[bytes, bytes], int] = {}
        
        # 反向索引，记录每个pair出现在哪些单词中
        self.pair_to_words: dict[tuple[bytes, bytes], set[bytes]] = {}
        # 最大堆，用于快速在pair_freqs中找到频率最高的pair
        self.freq_max_heap = []
        if vocab :
            self.token_vocab: dict[int, bytes] = vocab
        if merges:
            self.merges = merges
    def invert(self, b: bytes, length=20) -> bytes:
        if len(b) < length:
            b += bytes(length - len(b))  # 补 0
        inverted = bytes(255 - x for x in b)
        return inverted

    def _push_pair_to_heap(self, pair: tuple[bytes, bytes], freq: int) -> None:
        pair0 = self.invert(pair[0])
        pair1 = self.invert(pair[1])
        sort_key = (-freq,pair0,pair1)
        heapq.heappush(self.freq_max_heap, (sort_key, pair))
    
    def _pop_pair_from_heap(self) :
        """从最大堆中弹出频率最高的字节对"""
        while self.freq_max_heap:
            sort_key, pair = heapq.heappop(self.freq_max_heap)
            freq,_,_ = sort_key
            freq = -freq
            if pair in self.pair_freqs and self.pair_freqs[pair] == freq:
                return pair,sort_key
        raise ValueError("堆没有返回频率最大的字节对")
    
    def find_best_pair(self) -> tuple[bytes, bytes]:
        """找到频率最高的字节对"""
        return self._pop_pair_from_heap()
    
    def _update_pair_freqs(self, new_pair, old_pair, word, word_freq) -> None:
        # 添加 new_pair
        self.pair_to_words.setdefault(new_pair, set()).add(word)
        self.pair_freqs[new_pair] = self.pair_freqs.get(new_pair, 0) + word_freq
        # 一个new_pair可能被多次添加到堆中，但是应该问题不大
        self._push_pair_to_heap(new_pair, self.pair_freqs[new_pair])

        # 减少 old_pair
        if old_pair in self.pair_freqs:
            self.pair_freqs[old_pair] -= word_freq
            if self.pair_freqs[old_pair] <= 0:
                del self.pair_freqs[old_pair]
            else:
                # 如果old_pair仍然存在，更新最大堆
                self._push_pair_to_heap(old_pair, self.pair_freqs[old_pair])
        # 这里就不删除pair_to_words和freq_max_heap的对应项了，前者我们只关心我们要查的pair有就行，后者我们在取出时会检查
    
    def initialize_splits_and_pairs(self, word_freqs: Counter) -> None:
        for word, word_freq in word_freqs.items():
            # 初始化splits，将单词转换为字节序列
            self.splits[word] = [bytes([b]) for b in word]

            word_pieces = self.splits[word]
            if len(word_pieces) == 1:
                continue
            for j, pair in enumerate(zip(word_pieces[:-1], word_pieces[1:])):
                # 扫描每个单词的每个字节对，初始化pair_freqs
                self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + word_freq

                # 记录pair出现在哪些单词中，初始化反向索引pair_to_words
                if pair not in self.pair_to_words:
                    self.pair_to_words[pair] = set()
                self.pair_to_words[pair].add(word)
        # 初始化最大堆
        for pair, freq in self.pair_freqs.items():
            self._push_pair_to_heap(pair, freq)
            
    def update_splits_and_pairs(self, best_pair: tuple[bytes, bytes], new_token: bytes, word_freqs: Counter) -> None:
        """更新splits和pair_freqs"""
        # 哪些词包含best_pair，需要被更新
        # 直接从反向索引中获取
        affected_words = list(self.pair_to_words.get(best_pair, set()))

        # 更新splits
        for word in affected_words:
            word_freq = word_freqs[word]
            word_pieces = self.splits[word]
            i = 0
            while i < len(word_pieces) - 1:
                if word_pieces[i] == best_pair[0] and word_pieces[i + 1] == best_pair[1]:
                    word_pieces[i] = new_token
                    word_pieces.pop(i + 1)
                    if best_pair in self.pair_freqs:
                        del self.pair_freqs[best_pair]
                    if i > 0:
                        # 添加 A, BC ; 减少 A, B
                        new_pair_left = (word_pieces[i-1], new_token)
                        old_pair_left = (word_pieces[i-1], best_pair[0])
                        self._update_pair_freqs(new_pair_left, old_pair_left, word, word_freq)
                    if i < len(word_pieces) - 1:
                        # 添加 BC, D ; 减少 C, D
                        new_pair_right = (new_token, word_pieces[i+1])
                        old_pair_right = (best_pair[1], word_pieces[i+1])
                        self._update_pair_freqs(new_pair_right, old_pair_right, word, word_freq)
                else:
                    i += 1
                       
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    def read_corpus(self, input_path: str) -> Iterable[list[str]]:
        """读取语料并按特殊token进行分割"""
        with open(input_path, 'rb') as f:
            # 获取分块边界
            boundaries = self.find_chunk_boundaries(f, self.num_processes, "<|endoftext|>".encode('utf-8'))
            for start, end in tqdm(list(zip(boundaries[:-1], boundaries[1:])), desc="读取语料"):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                yield re.split(self.special_tokens_pattern, chunk)
    
   
    
    def add_special_tokens(self) -> None:
        """将特殊token添加到词汇表中"""
        for m, token in enumerate(self.special_tokens):
            # self.token_vocab[self.vocab_size - len(self.special_tokens) + m] = token.encode('utf-8')
            self.token_vocab[m] = token.encode('utf-8')
    
    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        word_freq = Counter()
        pool = multiprocessing.Pool(self.num_processes)
        results = []
        for docs in self.read_corpus(input_path):
            results.append(pool.apply_async(build_word_frequency, args=(docs,)))
            # batch_word_freq = self.build_word_frequency(docs)
            # word_freq.update(batch_word_freq)
        pool.close()
        pool.join()
        for res in tqdm(results,):
            word_freq.update(res.get())
        self.token_vocab = {i+len(self.special_tokens): bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        self.merges = []
        self.initialize_splits_and_pairs(word_freq)
        
        for num_merge in tqdm(range(num_merges),desc="merge"):
            if not self.pair_freqs:
                break
            
            best_pair,_ = self.find_best_pair()
            self.merges.append(best_pair)
            
            new_token = best_pair[0] + best_pair[1]
                           
            self.token_vocab[len(self.special_tokens) + 256 + num_merge] = new_token
            
            self.update_splits_and_pairs(best_pair, new_token, word_freq)
        # 添加特殊token到词汇表
        self.add_special_tokens()
        return self.token_vocab, self.merges
    
    
    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
    # 保存词汇表
        with open(vocab_filepath, 'wb') as f:
            buffer = io.BytesIO()
            buffer.write(len(self.token_vocab).to_bytes(4, byteorder='little'))
            for token_id, token in self.token_vocab.items():
                buffer.write(token_id.to_bytes(4, byteorder='little'))
                buffer.write(len(token).to_bytes(4, byteorder='little'))
                buffer.write(token)
            f.write(buffer.getvalue())

        # 保存合并规则
        with open(merges_filepath, 'wb') as f:
            buffer = io.BytesIO()
            buffer.write(len(self.merges).to_bytes(4, byteorder='little'))
            for first, second in self.merges:
                buffer.write(len(first).to_bytes(4, byteorder='little'))
                buffer.write(first)
                buffer.write(len(second).to_bytes(4, byteorder='little'))
                buffer.write(second)
            f.write(buffer.getvalue())
                
    
if __name__ == "__main__":
    "./data/TinyStoriesV2-GPT4-train.txt"
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    trainer = BPETrainer(vocab_size, special_tokens)
    token_vocab, merges = trainer.train(input_path)
    print(token_vocab)
    trainer.save("./workshop/TinyStories_train_10000_token_vocab.bin", "./workshop/TinyStories_train_10000_merges.bin")
    
    # input_path = "/inspire/hdd/project/embodied-multimodality/public/jzzhou/assignment1-basics/data/owt_train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    # trainer = BPETrainer(vocab_size, special_tokens)
    # token_vocab, merges = trainer.train(input_path)
    # print(token_vocab)
    # trainer.save("./workshop/owt_train_32000vocab.bin", "./workshop/owt_train_32000merges.bin")
   