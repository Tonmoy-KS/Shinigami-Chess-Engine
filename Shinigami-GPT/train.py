# train.py
import logging
import os
import time

import deepspeed
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from shinigami.model import LanguageModel
from shinigami.utils import MemmapDataset, prepare_data

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    deepspeed.init_distributed()
    rank = int(os.environ['RANK'])
    is_main_process = rank == 0
    
    log.info("Starting Shinigami-GPT Pre-training...")
    if is_main_process:
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    if is_main_process and cfg.wandb.log:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))

    torch.manual_seed(1337 + rank)
    bin_path = prepare_data(cfg.data.tokenizer_path, cfg.data.dataset_path, cfg.model.vocab_size)
    train_dataset = MemmapDataset(bin_path, cfg.model.block_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.micro_batch_size, sampler=DistributedSampler(train_dataset, shuffle=True))
    
    model = LanguageModel(cfg.model)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=cfg.deepspeed_config
    )

    log.info(f"Starting training on rank {rank}. Max iterations: {cfg.training.max_iters}")
    t0 = time.time()
    for iter_num in range(cfg.training.max_iters):
        engine.train()
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(engine.device), yb.to(engine.device)
        
        loss = engine(xb, yb)[1]
        engine.backward(loss)
        engine.step()

        if iter_num > 0 and iter_num % 10 == 0 and is_main_process:
            dt = time.time() - t0; t0 = time.time()
            lr = engine.get_lr()[0]; grad_norm = engine.get_global_grad_norm()
            log.info(f"Iter {iter_num}: Loss {loss.item():.4f}, LR {lr:.6f}, GradNorm: {grad_norm:.4f}, Time {dt*1000:.2f}ms")
            if cfg.wandb.log: wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": grad_norm})

    if is_main_process:
        log.info("Training finished. Saving final checkpoint.")
        save_dir = os.path.join(cfg.training.checkpoint_dir, cfg.wandb.run_name)
        engine.save_checkpoint(save_dir)

if __name__ == "__main__":
    main()