from dotenv import load_dotenv
import os
from dataclasses import dataclass
import torch

load_dotenv()

API_KEY = os.getenv("API_KEY", "arl-vlm")
API_KEY_NAME = "X-API-Key"

USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
print(f"Using CUDA: {USE_CUDA}")
print(f"Cuda available: {torch.cuda.is_available()}")
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


@dataclass
class DeepSeekVL2Config:
    """Configuration for Deepseek."""

    model_name: str = "deepseek-ai/deepseek-vl2"
    cropping: bool = False
    height: int = 512
    width: int = 512
