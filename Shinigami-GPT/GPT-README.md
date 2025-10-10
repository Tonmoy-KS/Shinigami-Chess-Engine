### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Shinigami-GPT
    ```
2.  **Install as a package:**
    Installing in editable mode (`-e`) allows you to modify the source code without reinstalling.
    ```bash
    pip install -e .
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Highly Recommended)** Install performance libraries for compatible GPUs:
    ```bash
    pip install flash-attn --no-build-isolation
    pip install triton # On Linux with NVIDIA GPU
    ```

### Workflow

#### Step 1: Prepare Data & Tokenizer

1.  Place your raw text data in a file (e.g., `input.txt`).
2.  Train a BPE tokenizer on your data. This will create `bpe_tokenizer.json`.
    ```bash
    python train_tokenizer.py
    ```

#### Step 2: Configure Your Run

All configurations are located in the `configs/` directory and managed by Hydra. The main entry point is `configs/config.yaml`.

-   **Model Architecture**: Edit `configs/model/main.yaml` or `draft.yaml`.
-   **Training Parameters**: Edit `configs/training/default.yaml`.
-   **DeepSpeed**: Choose your strategy in `configs/deepspeed/`.

You can override any parameter from the command line. For example, to use a different model or DeepSpeed config:
`deepspeed train.py model=draft deepspeed_config=configs/deepspeed/stage3.json`

#### Step 3: Pre-training a Base Model

Use `train.py` to pre-train your model on the raw text corpus. A `hostfile` is recommended for multi-GPU training.

-   **Create a `hostfile`:**
    ```
    # File named 'hostfile'
    localhost slots=4 # Replace 4 with your number of GPUs
    ```
-   **Launch Pre-training:**
    ```bash
    deepspeed --hostfile hostfile train.py
    ```
    Checkpoints will be saved in the directory specified in `configs/training/default.yaml`.

#### Step 4: Supervised Fine-Tuning (SFT)

Use `finetune_sft.py` to adapt your pre-trained model to an instruction-following format.

1.  **Prepare your SFT data:** Create a `.jsonl` file where each line is a JSON object, like:
    `{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}`
2.  **Update the data path** in `configs/data/default.yaml`.
3.  **Launch SFT:**
    You must specify the path to your pre-trained checkpoint.
    ```bash
    deepspeed --hostfile hostfile finetune_sft.py training.resume_from_ckpt=<path_to_pretrained_checkpoint_dir>
    ```

#### Step 5: Convert Checkpoint for Inference

DeepSpeed saves checkpoints in a sharded format. Convert them to a single file for easy inference.

```bash
python convert_deepspeed_checkpoint.py <path_to_deepspeed_checkpoint_dir> <output_model.pt>
# Example
python convert_deepspeed_checkpoint.py out/checkpoints/SFT-Run/latest out/sft_model.pt