# -*- coding: utf-8 -*-
# """
# visualize_data.py
# Created on September 3, 2025
# @ Author: Mazhar
# ""

from data_pipeline import view_dataset_instance

# Dataset Viewer
VIEW_SPLIT = "train"  # The split to view ('train' or 'validation').
VIEW_INDEX = 5  # The index of the example to view.

# Define the path to the configuration file
CONFIG_FILE_PATH = "fineTune/configs/configs.yaml"


# --- View a Dataset Instance ---
def main():
    print("🚀 Executing Dataset Viewer...")
    view_dataset_instance(
        config_path=CONFIG_FILE_PATH, split=VIEW_SPLIT, index=VIEW_INDEX
    )
    print("\nViewer finished.")
    return


if __name__ == "__main__":
    main()
