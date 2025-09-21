# -*- coding: utf-8 -*-
# """
# fineTune/utils/model_exporter.py
# Description: Utility functions for saving and exporting
# the final trained model.
# Author: Mazhar
# Created on: September 3, 2025
# """

import os

from .env_utils import get_hf_credentials
from .make_clean_dir import make_clean_dir


def save_merged_model_locally(model, processor, config: dict):
    # ... (This function remains unchanged) ...
    print("\n--- Merging adapters and saving the full model locally ---")
    try:
        merged_model_dir = config["fineTune"]["merged_model_output_dir"]
        make_clean_dir(merged_model_dir)
        model.save_pretrained_merged(merged_model_dir, tokenizer=processor)
        print(f"✅ Full, merged model saved to: {merged_model_dir}")
    except Exception as e:
        print(f"❌ ERROR during local model save: {e}")


def push_merged_model_to_hub(model, processor, config: dict):
    """
    Merges adapters and pushes the full model to the Hub, using credentials
    appropriate for the current environment (Colab or local).
    """
    print("\n--- Merging adapters and pushing to the Hugging Face Hub ---")

    # --- THIS IS THE NEW, SMART LOGIC ---
    # Get credentials from the correct source (Colab secrets or .env file)
    hf_username, hf_token = get_hf_credentials()

    # Stop if credentials were not found
    if not hf_username or not hf_token:
        print("❌ ERROR: Cannot push to Hub without credentials.")
        return
    # ------------------------------------

    try:
        export_config = config["fineTune"]["export"]
        # Get the repository name directly from the config file.
        repo_name = export_config.get("hub_repo_name")
        hub_model_name = f"{hf_username}/{repo_name}"

        # Pass the token explicitly to the push command
        print(f"Uploading to Hugging Face repository: {hub_model_name}")
        model.push_to_hub_merged(hub_model_name, tokenizer=processor, token=hf_token)

        print(
            f"✅ Full, merged model pushed to the Hub at: https://huggingface.co/{hub_model_name}"
        )

    except Exception as e:
        print(f"❌ ERROR during push to Hub: {e}")
