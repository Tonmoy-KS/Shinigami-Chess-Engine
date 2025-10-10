# evaluate.py
import argparse
import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from shinigami.model import LanguageModel
from shinigami.utils import MemmapDataset, prepare_data

log = logging.getLogger(__name__)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate Shinigami-GPT perplexity.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the converted model checkpoint (.pt file).')
    args = parser.parse_args()

    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    log.info(f"Loading model from {args.ckpt_path}...")
    model = LanguageModel(cfg.model)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    log.info("Loading validation data...")
    bin_path = prepare_data(cfg.data.tokenizer_path, cfg.data.dataset_path, cfg.model.vocab_size)
    val_dataset = MemmapDataset(bin_path, cfg.model.block_size)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.micro_batch_size)

    total_loss = 0.0
    num_batches = 0
    log.info("Evaluating perplexity on the validation set...")
    for xb, yb in tqdm(val_loader, desc="Validation"):
        if num_batches >= cfg.training.evaluation_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print("\n" + "="*30)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity.item():.2f}")
    print("="*30)

if __name__ == "__main__":
    main()