# -*- coding: utf-8 -*-
# """
# fineTune/data_pipeline/dataset_extraction.py
# Description:
# A single, organized class to handle the entire `VizWizDataPipeline` data preparation pipeline.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import json
import os
import shutil
import zipfile

import gdown
import requests
import yaml
from tqdm import tqdm


class VizWizDataPipeline:
    """
    A class to manage the download, extraction, and preparation of the
    VizWiz dataset for fine-tuning.
    """

    def __init__(self, config_path):
        """
        Initializes the pipeline by loading the configuration from a YAML file.
        """
        print("Initializing the data pipeline...")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # Determine which set of URLs to use
            self.use_gdrive_urls = self.config["dataset"].get("use_gdrive_urls", False)
            if self.use_gdrive_urls:
                self.urls = self.config["dataset"]["urls"]["gdrive_urls"]
                print("Using Google Drive URLs for dataset download.")
            else:
                self.urls = self.config["dataset"]["urls"]["original_urls"]
                print("Using Original Dataset URLs for dataset download.")

            self.paths = self.config["dataset"]["paths"]

            # Define key directories for easier access
            self.base_dir = self.paths["base_dir"]
            self.temp_zip_dir = self.paths["temp_zip_dir"]
            self.images_dir = self.paths["images_dir"]
            self.annotations_dir = self.paths["annotations_dir"]

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except KeyError as e:
            raise KeyError(f"Missing key in configuration file: {e}")

    def _download_and_unzip_smart(self, url, extract_dir, description):
        """
        Downloads a file and extracts its contents intelligently, avoiding the
        creation of nested directories (e.g., 'train/train').
        This method makes fix_paths.py obsolete.
        """
        os.makedirs(self.temp_zip_dir, exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)

        filename = url.split("/")[-1]
        zip_path = os.path.join(self.temp_zip_dir, filename)

        # --- Download Logic ---
        print(f"Downloading {description}...")

        if self.use_gdrive_urls:
            gdown.download(url, zip_path, quiet=False)
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=description
            ) as pbar:
                with open(zip_path, "wb") as f:
                    for data in response.iter_content(chunk_size=1024):
                        f.write(data)
                        pbar.update(len(data))

        # --- Smart Unzip Logic ---
        print(f"\nUnzipping {filename} directly to target location...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in tqdm(zf.infolist(), desc="Extracting"):
                if member.is_dir():
                    continue
                # This line strips the leading directory from the zip file
                target_filename = os.path.basename(member.filename)
                if not target_filename:
                    continue

                target_path = os.path.join(extract_dir, target_filename)
                with zf.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        print(f"Successfully extracted to: {extract_dir}")
        os.remove(zip_path)  # Clean up zip file
        print("-" * 30)

    def _prepare_annotations(self):
        """
        Processes the raw JSON annotation files into clean .jsonl files
        ready for the Hugging Face datasets library.
        """
        print("--- Starting Dataset Preparation (JSON to JSONL) ---")

        train_json = os.path.join(self.annotations_dir, "train.json")
        val_json = os.path.join(self.annotations_dir, "val.json")
        test_json = os.path.join(self.annotations_dir, "test.json")

        # Process train and validation files (with captions)
        for split in ["train", "val"]:
            json_path = os.path.join(self.annotations_dir, f"{split}.json")
            images_base_dir = os.path.join(self.images_dir, split)
            output_path = os.path.join(self.annotations_dir, f"{split}.jsonl")

            print(f"Processing {split} annotations...")
            with open(json_path, "r") as f:
                data = json.load(f)

            id_to_path = {
                img["id"]: os.path.join(images_base_dir, img["file_name"])
                for img in data["images"]
            }
            records = []
            for ann in data["annotations"]:
                if not ann.get("is_rejected", False):
                    records.append(
                        {
                            "image_path": id_to_path[ann["image_id"]],
                            "caption": ann["caption"],
                        }
                    )

            with open(output_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"Saved {len(records)} records to {output_path}")

        # Process test file (no captions)
        print("Processing test annotations...")
        with open(test_json, "r") as f:
            data = json.load(f)
        output_path = os.path.join(self.annotations_dir, "test.jsonl")
        records = [
            {"image_path": os.path.join(self.images_dir, "test", img["file_name"])}
            for img in data["images"]
        ]
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(records)} test records to {output_path}")

    def run(self):
        """
        Executes the entire data preparation pipeline in the correct order.
        """
        print("🚀 =============================================== 🚀")
        print("      Starting the Full VizWiz Data Pipeline")
        print("🚀 =============================================== 🚀")

        # --- Step 1: Download and Extract Data ---
        # Define paths for each component
        train_img_path = os.path.join(self.images_dir, "train")
        val_img_path = os.path.join(self.images_dir, "val")
        test_img_path = os.path.join(self.images_dir, "test")

        if not os.path.exists(os.path.join(self.annotations_dir, "train.json")):
            self._download_and_unzip_smart(
                self.urls["annotations"], self.annotations_dir, "Annotations"
            )
        else:
            print("Annotations already exist. Skipping.")

        if not os.path.exists(train_img_path):
            self._download_and_unzip_smart(
                self.urls["train_images"], train_img_path, "Train Images"
            )
        else:
            print("Train images already exist. Skipping.")

        if not os.path.exists(val_img_path):
            self._download_and_unzip_smart(
                self.urls["val_images"], val_img_path, "Validation Images"
            )
        else:
            print("Validation images already exist. Skipping.")

        if not os.path.exists(test_img_path):
            self._download_and_unzip_smart(
                self.urls["test_images"], test_img_path, "Test Images"
            )
        else:
            print("Test images already exist. Skipping.")

        # --- Step 2: Prepare Annotations ---
        self._prepare_annotations()

        print("\n✅ =============================================== ✅")
        print("      Data Pipeline Completed Successfully!")
        print(f"      Dataset is ready at: {os.path.abspath(self.base_dir)}")
        print("✅ =============================================== ✅")


# if __name__ == "__main__":
#     # Define the path to your configuration file
#     CONFIG_FILE_PATH = "fineTune/configs/configs.yaml"

#     # Create and run the pipeline
#     pipeline = VizWizDataPipeline(config_path=CONFIG_FILE_PATH)
#     pipeline.run()
