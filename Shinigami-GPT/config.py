# config.py
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class MOEConfig:
    num_experts: int
    num_experts_per_tok: int

@dataclass
class TrainingConfig:
    batch_size: int
    max_iters: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    clip_grad: float
    decay_lr: bool
    warmup_iters: int
    min_lr: float

@dataclass
class EvaluationConfig:
    interval: int
    iters: int

@dataclass
class InfraConfig:
    device: str
    ckpt_path: str
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
    moe: MOEConfig
    use_flash_attn: bool
    training: TrainingConfig
    evaluation: EvaluationConfig
    infra: InfraConfig
    wandb: WandbConfig
    data: DataConfig
    generation: GenerationConfig

def load_config(path: str) -> ModelConfig:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Helper to recursively convert dicts to dataclasses
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
