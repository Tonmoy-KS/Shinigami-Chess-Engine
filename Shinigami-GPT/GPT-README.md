### How to Run the Framework

1.  **Setup:**
    *   Ensure all dependencies from `requirements.txt` and `flash-attn` are installed.
    *   Run `wandb login` in your terminal.
    *   Place a text corpus in `input.txt`.

2.  **Train Tokenizer:**
    ```bash
    python train_tokenizer.py
    ```

3.  **Train the Models:**
    You must train both the main model and the smaller draft model for speculative decoding.
    *   **Train Main Model (using 2 GPUs):**
        ```bash
        torchrun --standalone --nproc_per_node=2 train.py
        ```
    *   **Train Draft Model (using 2 GPUs):**
        ```bash
        torchrun --standalone --nproc_per_node=2 train.py --draft
        ```

4.  **Generate Text:**
    After both models are trained and their checkpoints (`out/model.pt` and `out/draft_model.pt`) exist, run generation.
    ```bash
    python generate.py
    ```
    This will show a comparison between standard and the faster speculative decoding.

5.  **Evaluate the Main Model:**
    ```bash
    python evaluate.py
    ```

6.  **Export the Main Model to ONNX:**
    ```bash
    python export_onnx.py
    ```
