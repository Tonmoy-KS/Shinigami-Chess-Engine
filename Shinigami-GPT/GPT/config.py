# gpt/config.py
"""Dataclasses for Hydra configuration."""
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class MOEConfig:
    num_experts: int
    num_experts_per_tok: int

@dataclass
class RopeScalingConfig:
    type: str
    factor: float

@dataclass
class ModelConfig:
    name: str
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_heads: int
    n_embed: int
    dropout: float
    untie_weights: bool
    use_bias: bool
    sliding_window_size: int
    moe: MOEConfig
    rope_scaling: RopeScalingConfig

@dataclass
class DataConfig:
    tokenizer_path: str
    dataset_path: str
    sft_dataset_path: str

@dataclass
class TrainingConfig:
    global_batch_size: int
    micro_batch_size: int
    max_iters: int
    clip_grad: float
    evaluation_interval: int
    evaluation_iters: int
    checkpoint_dir: str
    resume_from_ckpt: Optional[str]

@dataclass
class WandbConfig:
    log: bool
    project: str
    run_name: str

@dataclass
class InfraConfig:
    compile_model: bool

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    speculative_k: int

@dataclass
class ShinigamiConfig:
    """Root configuration dataclass managed by Hydra."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    wandb: WandbConfig
    infra: InfraConfig
    generation: GenerationConfig
    deepspeed_config: str