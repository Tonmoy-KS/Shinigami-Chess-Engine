# generate.py
import argparse
import sys
import time
import logging

import hydra
import torch
from omegaconf import DictConfig
from tokenizers import Tokenizer

from shinigami.model import LanguageModel

log = logging.getLogger(__name__)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate text with Shinigami-GPT.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the converted model checkpoint (.pt file).')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='The prompt to start generation from.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens to generate.')
    parser.add_argument('--use_swa', action='store_true', help='Enable Sliding Window Attention for long context.')
    args = parser.parse_args()

    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model from {args.ckpt_path}...")
    model = LanguageModel(cfg.model)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
    prompt_ids = tokenizer.encode(args.prompt).ids
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model.setup_caches(max_batch_size=1, device=device, use_swa=args.use_swa)

    print(f"\nPrompt: {args.prompt}", end="", flush=True)
    generated_tokens = []
    
    # Process prompt to fill KV cache
    model(context) 
    
    cur_token = context[:, -1:]
    for _ in range(args.max_new_tokens):
        logits, _ = model(cur_token)
        logits = logits[:, -1, :] / cfg.generation.temperature
        
        v, _ = torch.topk(logits, min(cfg.generation.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        cur_token = torch.multinomial(probs, num_samples=1)
        
        token_id = cur_token.item()
        if token_id == tokenizer.token_to_id("[EOS]"): break
        
        decoded_token = tokenizer.decode([token_id])
        print(decoded_token, end="", flush=True)
        time.sleep(0.02)
        
    print("\n\n--- End of Generation ---")

if __name__ == "__main__":
    main()