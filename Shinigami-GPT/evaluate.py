# evaluate.py
import torch
from tqdm import tqdm

from config import load_config
from model import LanguageModel
from utils import create_dataloader, prepare_data

@torch.no_grad()
def run_eval():
    cfg = load_config("config.yaml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LanguageModel(cfg).to(device)
    state_dict = torch.load(cfg.infra.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    bin_path = prepare_data(cfg.data.tokenizer_path, cfg.data.dataset_path, cfg.vocab_size)
    # Note: Dataloader is not distributed here, evaluating on a single GPU
    val_loader = create_dataloader(bin_path, cfg.block_size, cfg.training.batch_size, 0, 1, 'val')

    print("Evaluating perplexity on the validation set...")
    total_loss = 0.0
    num_batches = 0
    for xb, yb in tqdm(val_loader, desc="Validation"):
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity.item():.2f}")

if __name__ == "__main__":
    run_eval()
