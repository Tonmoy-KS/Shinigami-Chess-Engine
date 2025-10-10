# gpt/utils.py
import json
import logging
import os

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

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

class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer: Tokenizer, block_size):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.tokenizer.encode(item['prompt']).ids
        completion = self.tokenizer.encode(item['completion']).ids
        
        full_ids = prompt + completion
        if len(full_ids) > self.block_size + 1:
            full_ids = full_ids[:self.block_size + 1]
        
        x = torch.full((self.block_size,), -1, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)
        
        x[:len(full_ids)-1] = torch.tensor(full_ids[:-1])
        y[:len(full_ids)-1] = torch.tensor(full_ids[1:])
        
        loss_mask = torch.zeros_like(x, dtype=torch.bool)
        loss_mask[len(prompt)-1 : len(full_ids)-1] = True
        
        return x, y, loss_mask

def prepare_data(tokenizer_path, dataset_path, vocab_size):
    bin_path = os.path.splitext(dataset_path)[0] + ".bin"
    if not os.path.exists(bin_path):
        log.info(f"Tokenizing '{dataset_path}' and saving to '{bin_path}'...")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please run train_tokenizer.py first.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
            
        tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(dataset_path, 'r', encoding='utf-8') as f: text = f.read()
        
        ids = tokenizer.encode(text).ids
        dtype = np.uint16 if vocab_size < 65535 else np.int32
        arr = np.array(ids, dtype=dtype)
        arr.tofile(bin_path)
    else:
        log.info(f"Found pre-tokenized data at '{bin_path}'.")
    return bin_path