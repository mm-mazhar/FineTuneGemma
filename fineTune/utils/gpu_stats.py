# -*- coding: utf-8 -*-
# """
# fineTune/utils/gpu_status.py
# Description:
# Utility functions for monitoring and reporting GPU status and memory usage.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import gc
import subprocess
from typing import Dict

import psutil
import torch


def get_gpu_status():
    """
    Provides a comprehensive status of the system's hardware, including
    RAM, GPU details from nvidia-smi, and PyTorch-specific GPU memory usage.
    """
    print("=" * 50)
    print("      System Hardware & Memory Status")
    print("=" * 50)

    # --- 1. System RAM ---
    print("\n--- RAM Status ---")
    ram_gb = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GiB
    print(f"Total System RAM: {ram_gb:.2f} GiB")
    if ram_gb < 20:
        print("Note: This is not a high-RAM runtime.")
    else:
        print("Note: You are using a high-RAM runtime!")

    # --- 2. GPU Status ---
    print("\n--- GPU Status ---")
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU detected by PyTorch.")
        print("=" * 50)
        return  # Exit the function if no GPU is found

    # a) Run nvidia-smi for a detailed hardware report
    print("\n--- nvidia-smi Report ---")
    try:
        # Use subprocess for a more robust way to run shell commands in Python
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Could not run 'nvidia-smi'. The command may not be in your system's PATH."
        )
        print("Skipping nvidia-smi report.")

    # b) PyTorch-specific memory details
    print("\n--- PyTorch GPU Details ---")

    # Clear cache for the most accurate report
    gc.collect()
    torch.cuda.empty_cache()

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs detected by PyTorch: {gpu_count}")

    for i in range(gpu_count):
        gpu_stats = torch.cuda.get_device_properties(i)
        total_mem_gb = round(gpu_stats.total_memory / (1024**3), 2)

        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        # Reserved memory is the total memory managed by PyTorch's caching allocator
        reserved_mem_gb = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
        # Allocated memory is the memory actively used by tensors
        allocated_mem_gb = round(torch.cuda.memory_allocated(i) / (1024**3), 2)

        print(f"GPU {i}: {gpu_stats.name}")
        print(f"    - Start GPU Memory: {start_gpu_memory} GB of memory reserved.")
        print(f"    - Max memory = {max_memory} GB.")
        # print(f"    - Total Memory:          {total_mem_gb} GiB")
        # print(f"    - PyTorch Reserved Memory: {reserved_mem_gb} GiB")
        # print(f"    - PyTorch Allocated Memory:  {allocated_mem_gb} GiB")

    print("\n" + "=" * 50)


def get_gpu_usage_stats(start_gpu_memory: float) -> Dict[str, float]:
    """
    Calculates GPU memory statistics after a training run.

    This function is safe to call even if no CUDA-enabled GPU is present.

    Args:
        start_gpu_memory (float): The peak GPU memory reserved (in GB)
                                  before the training process started.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "peak_memory_gb": Peak memory reserved by PyTorch during the run.
            - "training_memory_gb": Memory specifically used for training (peak - start).
            - "peak_memory_percent": Peak memory as a percentage of the GPU's total memory.
            - "training_memory_percent": Training memory as a percentage of total.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Returning zero for memory stats.")
        return {
            "peak_memory_gb": 0.0,
            "training_memory_gb": 0.0,
            "peak_memory_percent": 0.0,
            "training_memory_percent": 0.0,
        }

    # Get total memory of the GPU
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory_gb = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    # Get the peak memory reserved by PyTorch during the run
    peak_memory_gb = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

    # Calculate the memory used specifically for the training process
    training_memory_gb = round(peak_memory_gb - start_gpu_memory, 3)

    # Calculate percentages, handling the case of max_memory_gb being zero
    peak_memory_percent = (
        round((peak_memory_gb / max_memory_gb) * 100, 3) if max_memory_gb > 0 else 0.0
    )
    training_memory_percent = (
        round((training_memory_gb / max_memory_gb) * 100, 3)
        if max_memory_gb > 0
        else 0.0
    )

    return {
        "peak_memory_gb": peak_memory_gb,
        "training_memory_gb": training_memory_gb,
        "peak_memory_percent": peak_memory_percent,
        "training_memory_percent": training_memory_percent,
    }


# ... (your other utility functions like display_training_summary, etc.)

# # --- Example of how to use the function ---
# if __name__ == "__main__":
#     # This will run if you execute this script directly
#     get_gpu_status()
