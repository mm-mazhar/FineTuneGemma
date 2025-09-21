# -*- coding: utf-8 -*-
# """
# train.py
# Description
# This script runs the complete fine-tuning
# process for the Gemma model using the
# prepared VizWiz dataset.
# Author: Mazhar
# Created on: September 3, 2025
# """

# --- Unsloth must be imported before transformers, trl, peft ---
# isort: off
from unsloth import FastVisionModel  # FastLanguageModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import get_chat_template

# isort: on

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.cuda
import yaml
from data_pipeline import load_dataset
from trl import SFTConfig, SFTTrainer

# --- Local Utilities ---
from utils import (
    DetailedLoggingCallback,
    display_evaluation_summary,
    display_training_summary,
    get_gpu_usage_stats,
    make_clean_dir,
    push_merged_model_to_hub,
    save_merged_model_locally,
    setupTensorboard,
)

# --- Load Configuration ---
CONFIG_FILE_PATH = "fineTune/configs/configs.yaml"

try:
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"❌ ERROR: Failed to load configuration file '{CONFIG_FILE_PATH}': {e}")
    sys.exit(1)
# ------------------------------------------------------------------------------

# --- Define Constants ---
MODEL_ID = config["fineTune"]["model_to_use"]
PROCESSED_DATASET_PATH = config["dataset"]["paths"]["processed_dir"]
FINAL_ADAPTERS_OUTPUT_DIR = config["fineTune"]["adapters_output_dir"]
TRAINER_OUTPUT_DIR = config["fineTune"]["trainer_output_dir"]
LOGGING_DIR = config["fineTune"]["logging_dir"]

# Assume these are defined earlier in your script
start_gpu_memory: float = 0.0
max_memory: float = 1.0
# ------------------------------------------------------------------------------

# --- Prepare Directories ---
make_clean_dir(FINAL_ADAPTERS_OUTPUT_DIR)
make_clean_dir(TRAINER_OUTPUT_DIR)
make_clean_dir(LOGGING_DIR)
# ------------------------------------------------------------------------------

# --- Load Model and Processor ---
print(f"--- Loading model with Unsloth: {MODEL_ID} ---")
model, vision_processor = FastVisionModel.from_pretrained(
    model_name=MODEL_ID,
    load_in_4bit=True,  # We can safely re-enable 4-bit!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context,
    # max_seq_length=1024,  # You can define this here
    # dtype=torch.bfloat16,  # Use bfloat16
    # device_map="auto",
)
print("✅ Model and processor loaded and optimized by Unsloth.")
# ------------------------------------------------------------------------------

# --- Configure LoRA (PEFT Adapters) with Unsloth ---
# Unsloth patches the model to prepare it for LoRA
model = FastVisionModel.get_peft_model(
    model=model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,  # 0.05
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    target_modules="all-linear",  # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
print("✅ LoRA configured.")
# ------------------------------------------------------------------------------

# --- Load Dataset ---
dataset = load_dataset(config)
# ------------------------------------------------------------------------------

# --- Create custom logging callback ---
logging_callback = DetailedLoggingCallback(logging_dir=LOGGING_DIR)
# ------------------------------------------------------------------------------

# --- Configure Training using SFTConfig ---

# Enable for training!
FastVisionModel.for_training(model)

print("\n--- Configuring the SFTTrainer ---")
training_args = SFTConfig(
    output_dir=TRAINER_OUTPUT_DIR,
    per_device_train_batch_size=1,  # Increased from 1 to 4!
    gradient_accumulation_steps=4,  # Can be reduced to 4 (Effective batch size = 16, 8)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Modern way to configure
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,
    max_steps=30,
    # num_train_epochs=1, # Set this instead of max_steps for full training runs
    per_device_eval_batch_size=4,
    learning_rate=2e-4,  # 2e-4, 2e-5
    save_strategy="steps",  # epoch, steps
    eval_strategy="steps",  # no, epoch, steps
    logging_steps=10,  # 1, 10, 50
    logging_dir=LOGGING_DIR,  # Specify logging directory
    logging_first_step=True,  # Log the first step
    eval_steps=10,  # Evaluate every 50 steps (in addition to epoch)
    save_steps=30,  # Save checkpoint every 100 steps
    load_best_model_at_end=True,  # Load best model at end of training
    metric_for_best_model="eval_loss",  # Use eval loss to determine best model
    greater_is_better=False,  # Lower eval loss is better
    optim="adamw_torch_fused",  # Use the standard optimizer, adamw_8bit, adamw_torch, adamw_torch_fused
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    # bf16=True,
    push_to_hub=False,
    report_to=["tensorboard"],  # Use list format for multiple reporters
    # packing=False,
    # Additional logging configurations
    dataloader_pin_memory=False,  # Can help with performance
    dataset_text_field="",
    remove_unused_columns=False,  # Keep all columns for debugging
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=2048,  # 1024
)

# --- Chat Template ---
vision_processor = get_chat_template(vision_processor, "gemma-3")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    processing_class=vision_processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, vision_processor),
    callbacks=[logging_callback],  # Add custom logging callback
    args=training_args,
    # formatting_func=formatting_func,
)
print("✅ SFTTrainer configured with enhanced logging.")
# ------------------------------------------------------------------------------

# --- Start Training ---
start_gpu_memory: float = 0.0
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()  # Reset stats to get a clean measurement for the training phase
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

print("\n🚀 --- Starting the fine-tuning process --- 🚀")
print("-" * 47)
print(f"TensorBoard logs will be saved to: {LOGGING_DIR}")
print(f"Training checkpoints will be saved to: {TRAINER_OUTPUT_DIR}")

# Configure and setup TensorBoard based on config settings
setupTensorboard(config, LOGGING_DIR)

print("-" * 60)

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
training_start_time = time.time()
trainer_stats = trainer.train()
training_end_time = time.time()
training_duration = training_end_time - training_start_time

print(f"\n🏁 --- Fine-tuning complete --- 🏁")
print(
    f"Total training time: {training_duration/3600:.2f} hours ({training_duration/60:.1f} minutes)"
)

# --- Calculate and Display GPU Memory Usage (After training) ---
memory_stats = get_gpu_usage_stats(start_gpu_memory=start_gpu_memory)

print("\n--- GPU Memory Usage Summary ---")
# The function handles the check, but we can check again for a cleaner print message
if torch.cuda.is_available():
    print(f"Peak reserved memory = {memory_stats['peak_memory_gb']} GB.")
    print(
        f"Peak reserved memory for training = {memory_stats['training_memory_gb']} GB."
    )
    print(
        f"Peak reserved memory % of max memory = {memory_stats['peak_memory_percent']} %."
    )
    print(
        f"Peak reserved memory for training % of max memory = {memory_stats['training_memory_percent']} %."
    )
else:
    print("No GPU was used, so no memory stats were recorded.")

print("-" * 60)
# ------------------------------------------------------------------------------

# --- Analyze and Display Results ---
print("\n--- Training Performance Summary ---")
training_summary = display_training_summary(trainer_stats)
print(training_summary)

print("\n--- Evaluation Performance Summary ---")
evaluation_summary = display_evaluation_summary(trainer)
print(evaluation_summary)
# ------------------------------------------------------------------------------

# --- Save the Fine-Tuning Artifacts ---
print(
    f"\n--- Saving lightweight LoRA adapters and processor to: {FINAL_ADAPTERS_OUTPUT_DIR} ---"
)
# This saves the tiny adapter files (your "blueprint" for the changes)
trainer.save_model(FINAL_ADAPTERS_OUTPUT_DIR)
# This saves the processor files (tokenizer.json, preprocessor_config.json, etc.)
# into the exact same directory, making it a complete, self-contained model folder.
vision_processor.save_pretrained(FINAL_ADAPTERS_OUTPUT_DIR)
print(f"✅ Fine-tuned artifacts saved successfully to {FINAL_ADAPTERS_OUTPUT_DIR}")
# -----------------------------------------------------------------------------------

# --- Export Merged Model to Hugging Face Hub ---
# Get the export configuration from your YAML file
export_config = config.get("fineTune", {}).get("export", {})
# Conditionally push the full, merged model to the Hub
if export_config.get("push_to_hub", False):
    push_merged_model_to_hub(model, vision_processor, config)
# -----------------------------------------------------------------------------------
