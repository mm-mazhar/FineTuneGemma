import json
import os
import random
import shutil
import zipfile

# --- Configuration --- #
BASE_DIR_V1 = "extras/dataset/vizWiz_v1"
BASE_DIR_V2 = "extras/dataset/vizWiz_v2"
SUBSET_PERCENTAGE = 0.20  # 20% of the dataset

# Define paths for the original dataset (v1)
ANNOTATIONS_DIR_V1 = os.path.join(BASE_DIR_V1, "annotations")
IMAGES_DIR_V1 = os.path.join(BASE_DIR_V1, "images")

TRAIN_IMG_DIR_V1 = os.path.join(IMAGES_DIR_V1, "train")
VAL_IMG_DIR_V1 = os.path.join(IMAGES_DIR_V1, "val")
TEST_IMG_DIR_V1 = os.path.join(IMAGES_DIR_V1, "test")

# Define paths for the new subset dataset (v2)
ANNOTATIONS_DIR_V2 = os.path.join(BASE_DIR_V2, "annotations")
IMAGES_DIR_V2 = os.path.join(BASE_DIR_V2, "images")

TRAIN_IMG_DIR_V2 = os.path.join(IMAGES_DIR_V2, "train")
VAL_IMG_DIR_V2 = os.path.join(IMAGES_DIR_V2, "val")
TEST_IMG_DIR_V2 = os.path.join(IMAGES_DIR_V2, "test")


def create_directories():
    """Create necessary directories for vizWiz_v2."""
    os.makedirs(ANNOTATIONS_DIR_V2, exist_ok=True)
    os.makedirs(TRAIN_IMG_DIR_V2, exist_ok=True)
    os.makedirs(VAL_IMG_DIR_V2, exist_ok=True)
    os.makedirs(TEST_IMG_DIR_V2, exist_ok=True)
    print(f"Created directories for {BASE_DIR_V2}")


def process_split(split_name: str):
    """
    Processes a single split (train, val, test) to create a subset.
    """
    print(f"\n--- Processing {split_name} split ---")

    # Define input paths for the current split
    json_path_v1 = os.path.join(ANNOTATIONS_DIR_V1, f"{split_name}.json")
    images_dir_v1 = os.path.join(IMAGES_DIR_V1, split_name)

    # Define output paths for the current split
    json_path_v2 = os.path.join(ANNOTATIONS_DIR_V2, f"{split_name}.json")
    images_dir_v2 = os.path.join(IMAGES_DIR_V2, split_name)

    # Load original annotation data
    if not os.path.exists(json_path_v1):
        print(f"  ❌ Error: Annotation file not found for {split_name}: {json_path_v1}")
        return

    with open(json_path_v1, "r") as f:
        data_v1 = json.load(f)

    # Select a random subset of images
    original_images = data_v1["images"]
    num_original_images = len(original_images)
    num_subset_images = max(1, int(num_original_images * SUBSET_PERCENTAGE))
    selected_images = random.sample(original_images, num_subset_images)

    print(f"  Original {split_name} images: {num_original_images}")
    print(f"  Selected {split_name} images for subset: {num_subset_images}")

    # Create a mapping from original image ID to new image ID (if IDs change) and file_name
    selected_image_ids = {img["id"] for img in selected_images}
    new_image_id_map = {
        img["id"]: img["id"] for img in selected_images
    }  # Keep same IDs for simplicity

    # --- Copy selected images ---
    os.makedirs(images_dir_v2, exist_ok=True)
    for img_info in selected_images:
        src_path = os.path.join(images_dir_v1, img_info["file_name"])
        dst_path = os.path.join(images_dir_v2, img_info["file_name"])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"  ⚠️  Warning: Image not found: {src_path}")

    # --- Create new annotation data ---
    new_data_v2 = {
        "info": data_v1.get("info", {}),
        "licenses": data_v1.get("licenses", []),
        "images": selected_images,
        "annotations": [],  # Initialize empty, will be populated if annotations exist
    }

    if "annotations" in data_v1:
        original_annotations = data_v1["annotations"]
        selected_annotations = [
            ann for ann in original_annotations if ann["image_id"] in selected_image_ids
        ]
        new_data_v2["annotations"] = selected_annotations
        print(f"  Original {split_name} annotations: {len(original_annotations)}")
        print(
            f"  Selected {split_name} annotations for subset: {len(selected_annotations)}"
        )
    else:
        print(
            f"  No annotations found for {split_name} split. This is expected for test.json."
        )

    # Save new annotation file
    with open(json_path_v2, "w") as f:
        json.dump(new_data_v2, f, indent=4)
    print(f"  ✅ Saved subset annotation file to: {json_path_v2}")


def zip_dataset(output_base_name="vizWiz_v2"):
    """
    Zips the created vizWiz_v2 dataset folders individually.
    """
    print(f"\n--- Zipping the {BASE_DIR_V2} dataset ---")
    output_parent_dir = os.path.dirname(BASE_DIR_V2)
    os.makedirs(output_parent_dir, exist_ok=True)

    # 2. Create individual zips for annotations, train, val, test
    individual_folders_to_zip = [
        ("annotations", ANNOTATIONS_DIR_V2),
        ("train", TRAIN_IMG_DIR_V2),
        ("val", VAL_IMG_DIR_V2),
        ("test", TEST_IMG_DIR_V2),
    ]

    for zip_name_suffix, folder_path in individual_folders_to_zip:
        if os.path.exists(folder_path):
            individual_zip_path = os.path.join(
                output_parent_dir, f"{zip_name_suffix}.zip"
            )
            print(
                f"  Creating individual zip for {zip_name_suffix}: {individual_zip_path}"
            )
            shutil.make_archive(
                os.path.join(output_parent_dir, zip_name_suffix),
                "zip",
                root_dir=os.path.dirname(folder_path),
                base_dir=os.path.basename(folder_path),
            )
            print(f"  ✅ Individual zip created: {individual_zip_path}")
        else:
            print(f"  ⚠️  Warning: Folder not found, skipping zip for {folder_path}")

    print(f"  ✅ All zipping operations complete.")


def main():
    print("🚀 Starting VizWiz Dataset Subsetting Script 🚀")
    create_directories()

    for split in ["train", "val", "test"]:
        process_split(split)

    zip_dataset()
    print("\n✅ VizWiz Dataset Subsetting Complete!")


if __name__ == "__main__":
    main()
