# -*- coding: utf-8 -*-
# """
# fineTune/utils/logging_callback.py
# Description:
# Custom logging callback for detailed training monitoring and progress tracking.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import os
import time
from transformers import TrainerCallback


class DetailedLoggingCallback(TrainerCallback):
    """
    Custom callback for detailed logging of training metrics and progress.
    """

    def __init__(self, logging_dir: str):
        self.logging_dir = logging_dir
        self.start_time = None
        self.log_file = os.path.join(logging_dir, "training_detailed.log")

        # Ensure log file exists
        os.makedirs(logging_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        message = f"Training began with {state.max_steps} total steps across {args.num_train_epochs} epochs"
        print(f"📊 {message}")
        self._log_to_file(message)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Calculate elapsed time
            elapsed = time.time() - self.start_time if self.start_time else 0

            # Format log message
            step = state.global_step
            epoch = state.epoch

            message_parts = [f"Step {step} (Epoch {epoch:.2f})"]

            if "loss" in logs:
                message_parts.append(f"Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                message_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")
            if "learning_rate" in logs:
                message_parts.append(f"LR: {logs['learning_rate']:.2e}")

            message_parts.append(f"Time: {elapsed/60:.1f}min")

            message = " | ".join(message_parts)
            print(f"📈 {message}")
            self._log_to_file(f"{message} | Full logs: {logs}")

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            message = f"Evaluation at step {state.global_step}: {logs}"
            print(f"🔍 {message}")
            self._log_to_file(message)

    def on_save(self, args, state, control, **kwargs):
        message = f"Model checkpoint saved at step {state.global_step}"
        print(f"💾 {message}")
        self._log_to_file(message)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        message = (
            f"Training completed in {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)"
        )
        print(f"🏁 {message}")
        self._log_to_file(message)
        self._log_to_file("=" * 50)

    def _log_to_file(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
