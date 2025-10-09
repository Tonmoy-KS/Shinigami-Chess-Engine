### How to Run the Framework

1.  **Setup:**
    *   Ensure all dependencies from `requirements.txt` are installed. For best performance, install `flash-attn`.
    *   Run `wandb login` in your terminal if you wish to log results.
    *   Place a text corpus in `input.txt`.
    *   Configure your main model, draft model, and training parameters in `config.yaml`.

2.  **Train Tokenizer:**
    ```bash
    python train_tokenizer.py
    ```

3.  **Train the Models:**
    You must train both the main model and the smaller draft model for speculative decoding. The configurations for both are now in `config.yaml`.
    *   **Train Main Model (using 2 GPUs):**
        ```bash
        torchrun --standalone --nproc_per_node=2 train.py
        ```
    *   **Train Draft Model (using 2 GPUs):**
        ```bash
        torchrun --standalone --nproc_per_node=2 train.py --draft
        ```

4.  **Generate Text:**
    After both models are trained and their checkpoints exist at the paths specified in `config.yaml`, run generation.
    ```bash
    python generate.py
    ```
    This will show a comparison between standard autoregressive decoding and the faster speculative decoding.

5.  **Evaluate the Main Model:**
    ```bash
    python evaluate.py
    ```

6.  **Export the Main Model to ONNX:**
    ```bash
    python export_onnx.py
    ```