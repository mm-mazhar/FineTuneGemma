# -*- coding: utf-8 -*-
# """
# fineTune/utils/colab_utils.py
# Description:
# Google Colab specific utilities for TensorBoard integration and environment detection.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import os
from typing import Any


def is_colab_environment() -> bool:
    """
    Check if the code is running in Google Colab environment.

    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab

        return True
    except ImportError:
        return False


def setup_tensorboard_colab(log_dir: str) -> bool:
    """
    Set up and start TensorBoard in Google Colab environment.

    Args:
        log_dir (str): Directory containing TensorBoard logs

    Returns:
        bool: True if TensorBoard was successfully started, False otherwise
    """
    if not is_colab_environment():
        print("⚠️  Not running in Google Colab environment")
        return False

    try:
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Get IPython instance for magic commands
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is None:
            print("⚠️  Could not get IPython instance")
            return False

        print(f"📊 Setting up TensorBoard for Google Colab...")
        print(f"📁 Log directory: {log_dir}")

        # Load TensorBoard extension
        ipython.run_line_magic("load_ext", "tensorboard")
        print("✅ TensorBoard extension loaded")

        # Start TensorBoard
        ipython.run_line_magic("tensorboard", f"--logdir {log_dir}")
        print("✅ TensorBoard started! The interface should appear above.")

        return True

    except Exception as e:
        print(f"❌ Error starting TensorBoard in Colab: {e}")
        print("💡 Manual setup instructions:")
        print("   1. Run in a new cell: %load_ext tensorboard")
        print(f"   2. Run in a new cell: %tensorboard --logdir {log_dir}")
        return False


def print_colab_instructions(log_dir: str):
    """
    Print instructions for manually setting up TensorBoard in Colab.

    Args:
        log_dir (str): Directory containing TensorBoard logs
    """
    print("📋 Google Colab TensorBoard Setup:")
    print("   To manually start TensorBoard, run these commands in separate cells:")
    print()
    print("   Cell 1:")
    print("   %load_ext tensorboard")
    print()
    print("   Cell 2:")
    print(f"   %tensorboard --logdir {log_dir}")
    print()
    print("   The TensorBoard interface will appear inline in your notebook!")
    print("   📊 You can monitor training progress in real-time")


def setupTensorboard(config: dict[str, Any], log_dir: str) -> None:
    """
    Set up TensorBoard based on configuration settings.

    Args:
        config (dict[str, Any]): Complete configuration dictionary
        log_dir (str): Directory containing TensorBoard logs
    """
    # Get TensorBoard configuration with defaults
    tensorboard_config = config.get("tensorboard", {})
    auto_start = tensorboard_config.get("auto_start", False)
    show_instructions = tensorboard_config.get("show_instructions", True)

    print("📊 TensorBoard Configuration:")
    print(f"   - Auto Start: {auto_start}")
    print(f"   - Show Instructions: {show_instructions}")
    print("-" * 40)

    # Handle different environments
    if is_colab_environment():
        print("🔍 Google Colab environment detected!")

        if auto_start:
            print("🚀 Auto-starting TensorBoard for Colab...")
            if setup_tensorboard_colab(log_dir):
                print("✅ TensorBoard should be running above!")
            else:
                if show_instructions:
                    print_colab_instructions(log_dir)
        else:
            print("ℹ️  TensorBoard auto-start is disabled in config")
            if show_instructions:
                print_colab_instructions(log_dir)
    else:
        print("💻 Local environment detected.")

        if auto_start:
            print(
                "⚠️  Auto-start TensorBoard is enabled but not supported for local environments"
            )
            print("💡 Consider using the monitor script:")
            print("   python fineTune/monitor_training.py tensorboard")

        if show_instructions:
            print("📋 Local TensorBoard Setup:")
            print(f"   Run: tensorboard --logdir {log_dir}")
            print("   Then open: http://localhost:6006 in your browser")
            print("   Or use: python fineTune/monitor_training.py tensorboard")
