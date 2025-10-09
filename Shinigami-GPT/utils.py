# utils.py
import math
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import os
from tokenizers import Tokenizer

# --- Learning Rate Scheduler ---
def get_cosine_lr(it, max_iters, warmup_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# --- Data Loading for Large Datasets ---
class MemmapDataset(Dataset):
    def __init__(self, data_file, block_size, dtype=np.uint16):
        self.block_size = block_size
        self.data = np.memmap(data_file, dtype=dtype, mode='r')

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def prepare_data(tokenizer_path, dataset_path, vocab_size):
    """Tokenizes and memory-maps the dataset."""
    bin_path = os.path.splitext(dataset_path)[0] + ".bin"
    if not os.path.exists(bin_path):
        print(f"Tokenizing '{dataset_path}' and saving to '{bin_path}'...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        ids = tokenizer.encode(text).ids
        dtype = np.uint16 if vocab_size < 65535 else np.int32
        arr = np.array(ids, dtype=dtype)
        arr.tofile(bin_path)
    return bin_path

def create_dataloader(data_file, block_size, batch_size, rank=0, world_size=1):
    dataset = MemmapDataset(data_file, block_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)
