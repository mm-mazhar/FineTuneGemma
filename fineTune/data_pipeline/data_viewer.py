# -*- coding: utf-8 -*-
# """
# fine-data_viewer.py
# Description:
# Utility function to view an instance from a path-based dataset.
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import copy
import os
import pprint

import yaml
from datasets import load_from_disk
from IPython.display import display
from PIL import Image

# io is no longer needed since we are not handling bytes


def load_images_from_paths(instance: dict) -> dict:
    """
    Finds image string paths in a dataset instance and replaces them
    with loaded PIL Image objects. Modifies the instance in-place.
    """
    if "messages" not in instance:
        return instance

    for message in instance["messages"]:
        if "content" not in message or not isinstance(message["content"], list):
            continue

        for item in message["content"]:
            # Check if this part of the content is an image path (a string)
            if item.get("type") == "image" and isinstance(
                item.get("image"), str
            ):  # <-- CRITICAL CHANGE
                image_path = item["image"]
                if image_path:
                    try:
                        # 1. Open the image from the path
                        pil_image = Image.open(image_path).convert("RGB")
                        # 2. Replace the string path with the PIL object
                        item["image"] = pil_image
                    except FileNotFoundError:
                        print(f"⚠️ WARNING: Image file not found at path: {image_path}")
                        item["image"] = None  # Set to None on failure
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not open image from path. Error: {e}")
                        item["image"] = None
    return instance


def view_dataset_instance(config_path: str, split: str, index: int):
    """
    Loads a path-based dataset, loads the image for a specific instance,
    and displays the full, cleaned dictionary and the image itself.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        processed_dir = config["dataset"]["paths"]["processed_dir"]
    if not os.path.exists(processed_dir):
        print(f"❌ ERROR: Processed dataset directory not found at: {processed_dir}")
        return

    print(f"🔄 Loading lightweight dataset from '{processed_dir}'...")
    dataset = load_from_disk(processed_dir)

    # --- (Validation checks are the same) ---
    if split not in dataset:
        print(f"❌ ERROR: Split '{split}' not found. Available: {list(dataset.keys())}")
        return
    dataset_split = dataset[split]
    if not (0 <= index < len(dataset_split)):
        print(
            f"❌ ERROR: Index {index} is out of bounds for split '{split}' (size: {len(dataset_split)})."
        )
        return

    # 1. Load the raw instance (which contains string paths)
    raw_instance = dataset_split[index]

    # 2. Create a deep copy and load the images from their paths
    instance_for_viewing = load_images_from_paths(copy.deepcopy(raw_instance))

    print(
        "\n"
        + "=" * 60
        + f"\n    Displaying FULL instance [{index}] from split '{split}'\n"
        + "=" * 60
        + "\n"
    )

    # 3. Use pprint to print the dictionary with the nice <PIL...> object
    pprint.pprint(instance_for_viewing)

    # 4. Extract and display the image for convenience
    image_to_display = None
    for message in instance_for_viewing.get("messages", []):
        for content_item in message.get("content", []):
            if content_item.get("type") == "image" and isinstance(
                content_item.get("image"), Image.Image
            ):
                image_to_display = content_item["image"]
                break
        if image_to_display:
            break

    if image_to_display:
        print("\n🖼️  Image Preview:")
        if "COLAB_" in "".join(os.environ.keys()):
            w, h = image_to_display.size
            image_to_display = image_to_display.resize((w // 2, h // 2))
            display(image_to_display)
        else:
            # Save the image locally for a reliable viewing experience
            output_dir = "fineTune/output_images"
            os.makedirs(output_dir, exist_ok=True)
            image_filename = f"viewed_{split}_instance_{index}.png"
            image_save_path = os.path.join(output_dir, image_filename)
            image_to_display.save(image_save_path)
            print(f"✅ Image saved to: {os.path.abspath(image_save_path)}")
            try:
                image_to_display.show()
            except Exception:
                print(
                    "   (Could not open image automatically. Please open the file manually.)"
                )

    print("\n" + "=" * 60)


# if __name__ == '__main__':
#     CONFIG_PATH = "fineTune/configs/configs.yaml"
#     view_dataset_instance(config_path=CONFIG_PATH, split="train", index=5)
