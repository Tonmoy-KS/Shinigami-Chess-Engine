# train_tokenizer.py
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 10000
INPUT_FILE = "input.txt"
TOKENIZER_FILE = "bpe_tokenizer.json"

def train_bpe_tokenizer():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please create a text file named 'input.txt' with your training corpus.")
        return

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

    print("Training tokenizer...")
    tokenizer.train(files=[INPUT_FILE], trainer=trainer)
    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer with vocab size {VOCAB_SIZE} saved to '{TOKENIZER_FILE}'")

if __name__ == "__main__":
    train_bpe_tokenizer()