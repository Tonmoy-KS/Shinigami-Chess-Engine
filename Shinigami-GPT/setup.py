# setup.py
from setuptools import setup, find_packages

setup(
    name="shinigami-gpt",
    version="1.18.5",
    packages=find_packages(),
    description="The LLM framework for Shinigami chess engine.",
    author="Tonmoy-KS",
    install_requires=[
        "torch",
        "tokenizers",
        "transformers",
        "wandb",
        "einops",
        "pyyaml",
        "numpy",
        "deepspeed",
        "hydra-core",
        "omegaconf",
        "tqdm",
    ],
    python_requires='>=3.8',
)