# finetune_sft.py
import logging
import os
import time

import deepspeed
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, DistributedSampler

from shinigami.model import LanguageModel
from shinigami.utils import SFTDataset

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    deepspeed.init_distributed()
    rank = int(os.environ['RANK'])
    is_main_process = rank == 0
    cfg.wandb.run_name = f"SFT-{cfg.model.name}"
    
    log.info("Starting Shinigami-GPT Supervised Fine-Tuning...")
    if is_main_process:
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    if is_main_process and cfg.wandb.log:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))
        
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
    train_dataset = SFTDataset(cfg.data.sft_dataset_path, tokenizer, cfg.model.block_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.micro_batch_size, sampler=DistributedSampler(train_dataset, shuffle=True))
    
    model = LanguageModel(cfg.model)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=cfg.deepspeed_config
    )

    if cfg.training.resume_from_ckpt:
        log.info(f"Loading checkpoint from: {cfg.training.resume_from_ckpt}")
        engine.load_checkpoint(cfg.training.resume_from_ckpt, load_optimizer_states=False, load_lr_scheduler_states=False)
    else:
        log.warning("Starting SFT without a pre-trained checkpoint. This is unusual.")

    log.info(f"Starting SFT on rank {rank}. Max iterations: {cfg.training.max_iters}")
    t0 = time.time()
    for iter_num in range(cfg.training.max_iters):
        engine.train()
        xb, yb, loss_mask = next(iter(train_loader))
        xb, yb, loss_mask = xb.to(engine.device), yb.to(engine.device), loss_mask.to(engine.device)
        
        loss = engine(xb, targets=yb, loss_mask=loss_mask)[1]
        engine.backward(loss)
        engine.step()

        if iter_num > 0 and iter_num % 10 == 0 and is_main_process:
            dt = time.time() - t0; t0 = time.time()
            lr = engine.get_lr()[0]
            log.info(f"Iter {iter_num}: Loss {loss.item():.4f}, LR {lr:.6f}, Time {dt*1000:.2f}ms")
            if cfg.wandb.log: wandb.log({"sft/loss": loss.item(), "sft/lr": lr})

    if is_main_process:
        log.info("SFT finished. Saving final SFT checkpoint.")
        save_dir = os.path.join(cfg.training.checkpoint_dir, cfg.wandb.run_name)
        engine.save_checkpoint(save_dir)

if __name__ == "__main__":
    main()