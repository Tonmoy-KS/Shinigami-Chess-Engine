### How to Run the Framework

This framework is now powered by DeepSpeed ZeRO for maximum scalability.

1.  **Setup:**
    *   Install dependencies: `pip install -r requirements.txt`. This now includes `deepspeed` and `triton` (for compatible NVIDIA GPUs).
    *   Run `wandb login` in your terminal if you wish to log results.
    *   Place a text corpus in `input.txt`.
    *   Configure your models, training parameters, and DeepSpeed settings in `config.yaml`.

2.  **Train Tokenizer:**
    ```bash
    python train_tokenizer.py
    ```

3.  **Train the Models:**
    The training script is now launched using `deepspeed`. A `hostfile` is recommended for multi-GPU or multi-node training.

    *   **Create a `hostfile` (optional, for multi-GPU):**
        ```
        # A file named 'hostfile' with the following content:
        localhost slots=4 # Replace 4 with your number of GPUs
        ```
    *   **Launch Main Model Training:**
        ```bash
        deepspeed --hostfile hostfile train.py
        ```
    *   **Launch Draft Model Training:**
        ```bash
        deepspeed --hostfile hostfile train.py --draft
        ```
    *   **Resume Training:** DeepSpeed handles resuming automatically. Just point `deepspeed_config.json` to the checkpoint directory (tag).

4.  **Convert Checkpoint for Inference:**
    DeepSpeed saves sharded checkpoints. Use the provided script to create a single-file model for inference.
    ```bash
    python convert_deepspeed_checkpoint.py <path_to_deepspeed_checkpoint_dir> <output_model.pt>
    ```
    For example:
    ```bash
    python convert_deepspeed_checkpoint.py out/checkpoints/main-model/latest out/model.pt
    ```

5.  **Generate Text:**
    Run generation using the *converted* checkpoint.
    ```bash
    python generate.py --ckpt_path out/model.pt
    ```

6.  **Evaluate the Main Model:**
    Run evaluation using the *converted* checkpoint.
    ```bash
    python evaluate.py --ckpt_path out/model.pt
    ```