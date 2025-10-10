# convert_deepspeed_checkpoint.py
import argparse
import os
import torch
from collections import OrderedDict

def convert_deepspeed_checkpoint(checkpoint_dir, output_path):
    print(f"Converting DeepSpeed checkpoint from: {checkpoint_dir}")
    
    state_dict_path = os.path.join(checkpoint_dir, 'mp_rank_00_model_states.pt')
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Checkpoint file not found at {state_dict_path}. "
                                "Ensure you are pointing to a valid DeepSpeed checkpoint directory "
                                "(e.g., 'out/checkpoints/my-run/latest').")

    deepspeed_state = torch.load(state_dict_path, map_location='cpu')
    
    model_state_dict = deepspeed_state.get('module', None)
    if model_state_dict is None:
        raise KeyError("'module' key not found in the checkpoint. The checkpoint structure might be different.")
        
    final_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        if key.startswith('_forward_module.'): new_key = key[len('_forward_module.'):]
        elif key.startswith('module.'): new_key = key[len('module.'):]
        else: new_key = key
        final_state_dict[new_key] = value

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_state_dict, output_path)
    print(f"Successfully converted and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to a standard .pt file.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the DeepSpeed checkpoint directory (e.g., '.../latest').")
    parser.add_argument("output_path", type=str, help="Path to save the converted .pt file.")
    args = parser.parse_args()
    convert_deepspeed_checkpoint(args.checkpoint_dir, args.output_path)