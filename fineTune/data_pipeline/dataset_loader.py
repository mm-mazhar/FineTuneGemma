# -*- coding: utf-8 -*-
# """
# fineTune/data_pipeline/dataset_loader.py
# Description:
# Utility functions for loading and preparing datasets for training.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import os
from typing import Any

from datasets import DatasetDict, load_from_disk


def load_dataset(config: dict[str, Any]) -> DatasetDict:
    """
    Load dataset from disk and apply subset configuration if enabled.

    Args:
        config (dict[str, Any]): Complete configuration dictionary containing
                                 dataset path and subset settings

    Returns:
        DatasetDict: Loaded and processed dataset ready for training
    """
    # Get dataset path from config
    processed_dataset_path = config["dataset"]["paths"]["processed_dir"]

    print(f"\n--- Loading dataset from disk: {processed_dataset_path} ---")

    # Check if dataset directory exists
    if not os.path.exists(processed_dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {processed_dataset_path}")

    # Load the dataset
    dataset = load_from_disk(processed_dataset_path)

    # Apply subset configuration if enabled
    # dataset = _apply_subset_config(dataset, config)

    # Shuffle the dataset for better training
    dataset = dataset.shuffle(seed=42)

    print("\n✅ Dataset loaded and ready for training:")
    print(dataset)

    return dataset


def _apply_subset_config(dataset: DatasetDict, config: dict[str, Any]) -> DatasetDict:
    """
    Apply subset configuration to the dataset if enabled.

    Args:
        dataset (DatasetDict): Original dataset
        config (dict[str, Any]): Configuration dictionary

    Returns:
        DatasetDict: Dataset with subset applied if configured
    """
    # Safely get the subset configuration from the YAML file
    subset_config = config.get("dataset", {}).get("subset", {"enabled": False})

    if not subset_config.get("enabled", False):
        print("ℹ️  Using full dataset (subset mode disabled)")
        return dataset

    print("--- ❗ Subset Mode Enabled ---")

    # Get percentages from config, with a default of 100% if not specified
    train_percentage = subset_config.get("train_percentage", 100)
    val_percentage = subset_config.get("val_percentage", 100)

    # Validate percentages
    if not (0 < train_percentage <= 100) or not (0 < val_percentage <= 100):
        raise ValueError("Subset percentages must be between 1 and 100")

    # Calculate the number of samples to select
    original_train_size = len(dataset["train"])
    original_val_size = len(dataset["val"])

    num_train_samples = int(original_train_size * (train_percentage / 100))
    num_val_samples = int(original_val_size * (val_percentage / 100))

    print(
        f"Using {train_percentage}% of training data ({num_train_samples}/{original_train_size} examples)."
    )
    print(
        f"Using {val_percentage}% of validation data ({num_val_samples}/{original_val_size} examples)."
    )

    # Select the subset
    dataset["train"] = dataset["train"].select(range(num_train_samples))
    dataset["val"] = dataset["val"].select(range(num_val_samples))

    return dataset


def get_dataset_info(dataset: DatasetDict) -> dict[str, Any]:
    """
    Get information about the loaded dataset.

    Args:
        dataset (DatasetDict): The dataset to analyze

    Returns:
        dict[str, Any]: Dictionary containing dataset information
    """
    info = {
        "splits": list(dataset.keys()),
        "total_examples": sum(len(dataset[split]) for split in dataset.keys()),
        "split_sizes": {split: len(dataset[split]) for split in dataset.keys()},
    }

    # Get feature information from first split
    if dataset:
        first_split = list(dataset.keys())[0]
        info["features"] = list(dataset[first_split].features.keys())

    return info
