# -*- coding: utf-8 -*-
# """
# fineTune/data_pipeline/dataset_transformation.py
# Description:
# Contains functions for transforming the prepared data into a model-ready format.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import os

import yaml
from datasets import Features
from datasets import Image as ImageFeature
from datasets import List, Value, load_dataset, load_from_disk
from PIL import Image


def convert_to_conversation(example):
    """
    Creates a conversational dictionary, embedding the image FILE PATH as a string.
    """
    instructions = (
        "You are a helpful assistant for a visually impaired person. "
        "Your task is to describe the scene in the provided image clearly and "
        "concisely, focusing on potential obstacles or key objects."
    )
    # try:
    #     image_object = Image.open(example["image_path"])  # .convert("RGB")
    # except FileNotFoundError:
    #     return {"messages": None}
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {"type": "image", "image": example["image_path"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["caption"]},
            ],
        },
    ]
    return {"messages": conversation}


def transform_data_for_tuning(config_path):
    print("--- Starting Dataset Transformation ---")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["dataset"]["paths"]
    annotations_dir = paths["annotations_dir"]
    processed_dir = paths["processed_dir"]

    print(f"Loading .jsonl files from: {annotations_dir}")
    data_files = {
        "train": os.path.join(annotations_dir, "train.jsonl"),
        "val": os.path.join(annotations_dir, "val.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    print(f"Original dataset loaded: {dataset}")

    subset_config = config.get("dataset", {}).get("subset", {"enabled": False})
    if subset_config.get("enabled", False):
        print("\n--- ❗ Subset Mode is ENABLED ---")
        train_percentage = subset_config.get("train_percentage", 1)
        val_percentage = subset_config.get("val_percentage", 1)
        for split in ["train", "val"]:
            percentage = train_percentage if split == "train" else val_percentage
            original_size = len(dataset[split])
            num_samples = int(original_size * (percentage / 100))
            dataset[split] = dataset[split].select(range(num_samples))
            print(
                f"  - Subsetting '{split}' split to {percentage}% ({num_samples}/{original_size} examples)."
            )
        print(f"\nNew dataset size for processing: {dataset}")
    else:
        print("\n--- Full dataset processing is enabled ---")

    print("\nApplying conversational transformation using .map()...")

    features = Features(
        {
            "messages": List(
                {
                    "role": Value("string"),
                    "content": List(
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                            "image": Value("string"),
                        }
                    ),
                }
            )
        }
    )

    transformed_dataset = dataset.map(
        convert_to_conversation,
        remove_columns=["image_path", "caption"],
        batched=False,
        features=features,  # Apply the schema
        load_from_cache_file=False,
    )

    original_sizes = {split: len(dataset[split]) for split in dataset.keys()}
    transformed_dataset = transformed_dataset.filter(
        lambda x: x["messages"] is not None
    )
    filtered_sizes = {
        split: len(transformed_dataset[split]) for split in transformed_dataset.keys()
    }
    print("\nTransformation complete. Dataset stats:")
    for split in original_sizes:
        print(
            f"  - {split}: {original_sizes[split]} -> {filtered_sizes[split]} samples"
        )
    print(f"\nSaving transformed dataset to: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    transformed_dataset.save_to_disk(processed_dir)
    print(
        f"✅ Your model-ready dataset is now saved at: {os.path.abspath(processed_dir)}"
    )


def verify_dataset_integrity(config_path: str):
    print("--- Starting Dataset Integrity Verification ---")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["dataset"]["paths"]
    annotations_dir = paths["annotations_dir"]
    processed_dir = paths["processed_dir"]
    subset_config = config.get("dataset", {}).get("subset", {"enabled": False})

    all_splits_valid = True
    processed_dataset = load_from_disk(processed_dir)

    for split in ["train", "val"]:
        print(f"\nVerifying '{split}' split...")
        processed_example_count = len(processed_dataset[split])

        jsonl_path = os.path.join(annotations_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"⚠️  WARNING: Source file '{jsonl_path}' not found.")
            all_splits_valid = False
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            original_line_count = sum(1 for _ in f)

        if subset_config.get("enabled", False):
            percentage = subset_config.get(f"{split}_percentage", 1)
            expected_count = int(original_line_count * (percentage / 100))
            print(f"  - Subset mode: expecting {expected_count} examples.")
        else:
            expected_count = original_line_count
            print(f"  - Full mode: expecting {expected_count} examples.")

        print(f"  - Examples in processed '{split}' set: {processed_example_count}")
        if expected_count == processed_example_count:
            print("  ✅ Verification successful for this split.")
        else:
            print("  ❌ VERIFICATION FAILED. The number of examples does not match.")
            all_splits_valid = False
    return all_splits_valid
