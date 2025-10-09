# config.py
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class MOEConfig:
    num_experts: int
    num_experts_per_tok: int

@dataclass
class RopeScalingConfig:
    type: str
    factor: float

@dataclass
class DraftModelConfig:
    tag: str
    n_layer: int
    n_head: int
    n_kv_heads: int
    n_embed: int

@dataclass
class TrainingConfig:
    global_batch_size: int
    micro_batch_size: int
    max_iters: int
    clip_grad: float

@dataclass
class EvaluationConfig:
    interval: int
    iters: int

@dataclass
class InfraConfig:
    checkpoint_dir: str
    use_amp: bool
    compile_model: bool

@dataclass
class WandbConfig:
    log: bool
    project: str
    run_name: str
    
@dataclass
class DataConfig:
    tokenizer_path: str
    dataset_path: str

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    speculative_k: int

@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_heads: int
    n_embed: int
    dropout: float
    untie_weights: bool
    use_bias: bool
    moe: MOEConfig
    rope_scaling: RopeScalingConfig
    use_flash_attn: bool
    draft_model: DraftModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    infra: InfraConfig
    wandb: WandbConfig
    data: DataConfig
    generation: GenerationConfig

def load_config(path: str) -> ModelConfig:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    def _dict_to_dataclass(data_class, data_dict):
        instance_fields = {}
        for name, f_type in data_class.__annotations__.items():
            if name in data_dict:
                val = data_dict[name]
                if hasattr(f_type, '__dataclass_fields__'):
                    instance_fields[name] = _dict_to_dataclass(f_type, val)
                else:
                    instance_fields[name] = val
        return data_class(**instance_fields)

    return _dict_to_dataclass(ModelConfig, cfg_dict)

def get_deepspeed_config(cfg: ModelConfig) -> Dict:
    """Generates a DeepSpeed config dictionary."""
    return {
        "train_global_batch_size": cfg.training.global_batch_size,
        "train_micro_batch_size_per_gpu": cfg.training.micro_batch_size,
        "gradient_accumulation_steps": "auto",
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1.0e-4, # Placeholder, will be overwritten by scheduler
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 3.0e-5,
                "warmup_max_lr": 3.0e-4,
                "warmup_num_steps": 200,
                "total_num_steps": cfg.training.max_iters
            }
        },
        "gradient_clipping": cfg.training.clip_grad,
        "bf16": {
            "enabled": cfg.infra.use_amp
        },
        "zero_optimization": {
            "stage": 2, # Stage 2 is a good balance of performance and memory savings
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
        "steps_per_print": 10,
    }