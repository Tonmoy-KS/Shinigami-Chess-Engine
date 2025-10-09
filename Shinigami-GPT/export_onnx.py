# export_onnx.py
import torch
from config import load_config
from model import LanguageModel

def main():
    cfg = load_config("config.yaml")
    device = 'cpu' # ONNX export is easiest on CPU
    
    # Disable flash attention for ONNX export as it's a custom kernel
    cfg.use_flash_attn = False
    
    model = LanguageModel(cfg).to(device)
    state_dict = torch.load(cfg.infra.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randint(0, cfg.vocab_size, (1, 128), dtype=torch.long) # Use a smaller block size
    onnx_path = "out/model.onnx"
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=['input_ids'],
        output_names=['logits', 'loss'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}},
        opset_version=14
    )
    print("Export complete.")

if __name__ == "__main__":
    main()
