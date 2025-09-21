#!/usr/bin/env python3
# """
# monitor_training.py
# A utility script to monitor fine-tuning progress and logs.
# Created on: September 3, 2025
# Author: Mazhar
# """

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config():
    """Load configuration from YAML file."""
    config_path = "fineTune/configs/configs.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load config: {e}")
        sys.exit(1)


def check_log_directory(log_dir):
    """Check if log directory exists and contains files."""
    if not os.path.exists(log_dir):
        print(f"❌ Log directory doesn't exist: {log_dir}")
        return False

    log_files = list(Path(log_dir).glob("*"))
    if not log_files:
        print(f"⚠️  Log directory is empty: {log_dir}")
        return False

    print(f"✅ Found {len(log_files)} files in log directory:")
    for file in log_files:
        print(f"   - {file.name}")
    return True


def tail_log_file(log_file, lines=20):
    """Display the last N lines of a log file."""
    if not os.path.exists(log_file):
        print(f"❌ Log file doesn't exist: {log_file}")
        return

    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]

        print(f"\n📋 Last {len(recent_lines)} lines from {os.path.basename(log_file)}:")
        print("-" * 60)
        for line in recent_lines:
            print(line.rstrip())
        print("-" * 60)
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def start_tensorboard(log_dir):
    """Start TensorBoard server."""
    try:
        print(f"🚀 Starting TensorBoard for {log_dir}...")
        print("Access TensorBoard at: http://localhost:6006")
        print("Press Ctrl+C to stop TensorBoard")

        # Start TensorBoard
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                log_dir,
                "--port",
                "6006",
            ]
        )
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard stopped.")
    except Exception as e:
        print(f"❌ Error starting TensorBoard: {e}")
        print("Make sure TensorBoard is installed: pip install tensorboard")


def monitor_training_progress(log_dir):
    """Monitor training progress by watching log files."""
    detailed_log = os.path.join(log_dir, "training_detailed.log")

    print("🔍 Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")

    try:
        last_size = 0
        while True:
            if os.path.exists(detailed_log):
                current_size = os.path.getsize(detailed_log)
                if current_size > last_size:
                    # New content added
                    with open(detailed_log, "r") as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content.strip():
                            print(new_content, end="")
                    last_size = current_size

            time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")
    except Exception as e:
        print(f"❌ Error monitoring: {e}")


def main():
    """Main function to handle different monitoring options."""
    config = load_config()
    log_dir = config["fineTune"]["logging_dir"]

    print("🔧 Fine-Tuning Training Monitor")
    print("=" * 40)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python monitor_training.py check     - Check log directory status")
        print("  python monitor_training.py logs      - Show recent log entries")
        print("  python monitor_training.py monitor   - Monitor training in real-time")
        print("  python monitor_training.py tensorboard - Start TensorBoard")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "check":
        check_log_directory(log_dir)

    elif command == "logs":
        detailed_log = os.path.join(log_dir, "training_detailed.log")
        tail_log_file(detailed_log)

    elif command == "monitor":
        monitor_training_progress(log_dir)

    elif command == "tensorboard":
        if check_log_directory(log_dir):
            start_tensorboard(log_dir)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Usage #
# ------------------------------------------------------------------

# In a Colab cell:
# Option 1: !python ./fineTune/monitor_training.py tensorboard
# Option 2:
# %load_ext tensorboard
# %tensorboard --logdir fineTune/logs
# -------------------------------------------------------------------

# Local
# -------------------------------------------------------------------
# Option 1:
# uv run python ./fineTune/monitor_training.py tensorboard
# Option 2:
# In a separate terminal window, run:
# tensorboard --logdir fineTune/logs
# Then open your browser to: http://localhost:6006
