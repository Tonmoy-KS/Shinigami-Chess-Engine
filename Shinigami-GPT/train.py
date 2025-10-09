# train.py
import functools
import os
import time

import torch
import torch.distributed as dist
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType, FullStateDictConfig

from config import ModelConfig, load_config
from model import LanguageModel, TransformerBlock
from utils import create_dataloader, get_cosine_lr, prepare_data

def setup_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_distributed():
    dist.destroy_process_group()

def get_model(cfg, device):
    if cfg.draft_model: # Training a draft model
      draft_config_dict = {
          'block_size': cfg.block_size,
          'vocab_size': cfg.vocab_size,
          'n_layer': cfg.draft_model.n_layer,
          'n_head': cfg.draft_model.n_head,
          'n_kv_heads': cfg.draft_model.n_kv_heads,
          'n_embed': cfg.draft_model.n_embed,
          'dropout': cfg.dropout,
          'untie_weights': cfg.untie_weights,
          'use_flash_attn': cfg.use_flash_attn,
          'moe': None
      }
      return LanguageModel(draft_config_dict).to(device)
    else: # Training the main model
      return LanguageModel(cfg).to(device)

def main(is_draft=False):
    cfg = load_config("config.yaml")
    
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = rank == 0
    device = f'cuda:{rank}'
    
    if is_main_process:
        os.makedirs("out", exist_ok=True)
        if cfg.wandb.log:
            run_name = cfg.wandb.run_name + ("-draft" if is_draft else "")
            wandb.init(project=cfg.wandb.project, name=run_name, config=vars(cfg))

    torch.manual_seed(1337 + rank)
    bin_path = prepare_data(cfg.data.tokenizer_path, cfg.data.dataset_path, cfg.vocab_size)
    train_loader = create_dataloader(bin_path, cfg.block_size, cfg.training.batch_size, rank, world_size, 'train')
    val_loader = create_dataloader(bin_path, cfg.block_size, cfg.training.batch_size, rank, world_size, 'val')

    # FSDP requires bfloat16 for optimal performance
    amp_dtype = torch.bfloat16 if cfg.infra.use_amp else torch.float32

    model_args = {'draft_model': True} if is_draft else {}
    model = get_model(cfg, device, **model_args)

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
        ),
        use_orig_params=True
    )

    if cfg.infra.compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, 
                                  weight_decay=cfg.training.weight_decay, 
                                  betas=(cfg.training.beta1, cfg.training.beta2))
    
    iter_num = 0
    t0 = time.time()
    for iter_num in range(cfg.training.max_iters):
        lr = get_cosine_lr(iter_num, cfg.training.max_iters, cfg.training.warmup_iters, 
                           cfg.training.learning_rate, cfg.training.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.training.grad_accum_steps):
            xb, yb = next(iter(train_loader))
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, loss = model(xb, yb)
                loss = loss / cfg.training.grad_accum_steps
            loss.backward()

        optimizer.step()

        if iter_num % cfg.evaluation.interval == 0 and is_main_process:
            dt = time.time() - t0
            t0 = time.time()
            print(f"Iter {iter_num}: Loss {loss.item() * cfg.training.grad_accum_steps:.4f}, Time {dt*1000:.2f}ms, LR {lr:.6f}")
            if cfg.wandb.log:
                wandb.log({"train_loss": loss.item() * cfg.training.grad_accum_steps, "lr": lr})

    if is_main_process:
        print("Training finished.")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = model.state_dict()
        
        save_path = cfg.draft_model.ckpt_path if is_draft else cfg.infra.ckpt_path
        torch.save(cpu_state_dict, save_path)
        print(f"Model saved to {save_path}")
        
    cleanup_distributed()

if __name__ == "__main__":
    import sys
    is_draft_training = '--draft' in sys.argv
    main(is_draft=is_draft_training)
