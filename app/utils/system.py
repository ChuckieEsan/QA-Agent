import torch
import os

def get_device() -> str:
    """自动检测计算设备"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"