# -*- coding: utf-8 -*-
# """
# fineTune/utils/env_utils.py
# Description:
# Utilities for handling different environments (local vs. Colab).
# Author: Mazhar
# Created on: September 3, 2025
# """

import os
from typing import Optional

from .colab_utils import is_colab_environment


def get_hf_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Safely retrieves Hugging Face credentials (username and token) based on
    the execution environment.

    In Colab, it reads from `userdata` (secrets).
    Locally, it reads from a `.env` file.

    Returns:
        A tuple containing (hf_username, hf_token).
        Returns (None, None) if credentials are not found.
    """
    hf_username = None
    hf_token = None

    if is_colab_environment():
        print(
            "INFO: Running in Colab environment. Trying to get secrets from userdata."
        )
        try:
            from google.colab import userdata

            hf_username = userdata.get("HF_USERNAME")
            hf_token = userdata.get("HF_TOKEN")
            if not hf_username or not hf_token:
                print("❌ WARNING: HF_USERNAME or HF_TOKEN not found in Colab secrets.")
                return None, None
        except ImportError:
            print("❌ ERROR: Could not import Colab libraries.")
            return None, None
    else:
        print(
            "INFO: Running in a local environment. Trying to get secrets from .env file."
        )
        # print(f"Colab Environment: {is_colab_environment()}")
        try:
            from dotenv import load_dotenv

            load_dotenv()
            hf_username = os.getenv("HF_USERNAME")
            hf_token = os.getenv("HF_TOKEN")
            if not hf_username or not hf_token:
                print(
                    "❌ WARNING: HF_USERNAME or HF_TOKEN not found in your .env file."
                )
                return None, None
        except ImportError:
            print("❌ ERROR: python-dotenv is not installed. Cannot load .env file.")
            return None, None

    print("✅ Hugging Face credentials loaded successfully.")
    return hf_username, hf_token
