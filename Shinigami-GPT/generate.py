# generate.py
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig, load_config
from model import LanguageModel

def load_model(config_obj, ckpt_path, device):
    """Loads a model with a given config and checkpoint path."""
    model = LanguageModel(config_obj)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def sample_top_k(logits, top_k):
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('inf')
    return F.softmax(logits, dim=-1)

@torch.no_grad()
def speculative_decode(target_model, draft_model, prompt, max_new_tokens, k, temperature, top_k):
    target_model.setup_kv_cache(1, target_model.cfg.block_size, prompt.device, target_model.tok_embeddings.weight.dtype)
    draft_model.setup_kv_cache(1, draft_model.cfg.block_size, prompt.device, draft_model.tok_embeddings.weight.dtype)
    
    # Process the prompt through both models
    target_model(prompt, use_kv_cache=True)
    draft_model(prompt, use_kv_cache=True)
    
    generated_tokens = prompt
    
    for _ in range(max_new_tokens):
        # 1. Draft k tokens from the draft model
        draft_tokens = draft_model.generate(generated_tokens[:, -1:], k, temperature, top_k)
        
        # 2. Get target model logits for the draft + original sequence
        target_logits, _ = target_model(draft_tokens, use_kv_cache=True)
        target_logits = target_logits[:, -k-1:, :] # Get logits for the new tokens
        
        # 3. Rejection sampling
        accepted_any = False
        for i in range(k):
            draft_token_id = draft_tokens[:, i+1]
            draft_logits, _ = draft_model(draft_tokens[:, i:i+1], use_kv_cache=True)
            draft_probs = sample_top_k(draft_logits / temperature, top_k)
            
            target_probs = sample_top_k(target_logits[:, i, :] / temperature, top_k)
            
            p = target_probs[0, draft_token_id]
            q = draft_probs[0, 0, draft_token_id]
            
            if torch.rand(1).item() < (p / q): # Accept
                generated_tokens = torch.cat([generated_tokens, draft_token_id.unsqueeze(0)], dim=1)
                accepted_any = True
            else: # Reject
                # Sample from the corrected distribution
                new_probs = (target_probs - draft_probs).clamp(min=0)
                new_probs /= new_probs.sum()
                next_tok = torch.multinomial(new_probs, num_samples=1)
                generated_tokens = torch.cat([generated_tokens, next_tok], dim=1)
                break
        
        # If all draft tokens were accepted, sample one more from the target
        if accepted_any and generated_tokens.size(1) == prompt.size(1) + k:
            final_logits = target_logits[:, -1, :]
            final_probs = sample_top_k(final_logits / temperature, top_k)
            next_tok = torch.multinomial(final_probs, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_tok], dim=1)

    return generated_tokens

def main():
    cfg = load_config("config.yaml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading target model...")
    target_model = load_model(cfg, cfg.infra.ckpt_path, device)
    
    print("Loading draft model...")
    draft_cfg = {
        'block_size': cfg.block_size, 'vocab_size': cfg.vocab_size, 'n_layer': cfg.draft_model.n_layer,
        'n_head': cfg.draft_model.n_head, 'n_kv_heads': cfg.draft_model.n_kv_heads,
        'n_embed': cfg.draft_model.n_embed, 'dropout': cfg.dropout, 'untie_weights': cfg.untie_weights,
        'use_flash_attn': cfg.use_flash_attn, 'moe': None
    }
    draft_model = load_model(draft_cfg, cfg.draft_model.ckpt_path, device)
    
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
    
    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains."
    prompt_ids = tokenizer.encode(prompt).ids
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # --- Standard Generation ---
    print("\n--- Standard Autoregressive Decoding ---")
    t0 = time.time()
    output_standard = target_model.generate(context, 100, cfg.generation.temperature, cfg.generation.top_k)
    t1 = time.time()
    print(tokenizer.decode(output_standard[0].tolist()))
    print(f"Time taken: {t1-t0:.2f}s")
    
    # --- Speculative Decoding ---
    print("\n--- Speculative Decoding ---")
    t0 = time.time()
    output_speculative = speculative_decode(target_model, draft_model, context, 100, 
                                            cfg.generation.speculative_k, 
                                            cfg.generation.temperature, 
                                            cfg.generation.top_k)
    t1 = time.time()
    print(tokenizer.decode(output_speculative[0].tolist()))
    print(f"Time taken: {t1-t0:.2f}s")

if __name__ == "__main__":
    main()
