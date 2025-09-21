# -*- coding: utf-8 -*-
# """
# fineTune/utils/capture_key_libs.py
# Description:
# Utility functions for capturing and logging key library versions.
# Created on September 3, 2025
# @ Author: Mazhar
# """


import importlib
import sys


def capture_key_dependency_versions():
    """
    Checks for key ML libraries in the environment, captures their
    exact versions, and prints them in a requirements.txt format.
    """
    print("=" * 50)
    print("   Capturing Key Dependency Versions for Reproducibility")
    print("=" * 50)

    # --- Define the key packages you want to track ---
    # These are the most critical libraries for your training pipeline.
    # The name here is the name you would use to `import` the package.
    packages_to_check = [
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "unsloth",
        "timm",
        "xformers",
        "pandas",
        "yaml",  # from pyyaml
        "triton",  # from triton
        "cut_cross_entropy",  # from cut_cross_entropy
        "unsloth_zoo",  # from unsloth_zoo
        "hf_transfer",  # from hf_transfer
        "huggingface_hub",  # from huggingface_hub
        "sentencepiece",  # from sentencepiece
        "protobuf",  # from protobuf
    ]

    requirements_list = []

    for package_name in packages_to_check:
        try:
            # Dynamically import the module
            module = importlib.import_module(package_name)

            # Get the version, default to 'N/A' if not found
            version = getattr(module, "__version__", "N/A")

            # Format for requirements.txt
            if version != "N/A":
                line = f"{package_name}=={version}"
                requirements_list.append(line)
                print(f"✅ Found: {line}")
            else:
                print(f"⚠️  Warning: Could not determine version for '{package_name}'.")

        except ImportError:
            print(f"❌ Error: Package '{package_name}' is not installed.")
        except Exception as e:
            print(f"An unexpected error occurred with {package_name}: {e}")

    print("\n--- Frozen Requirements ---")
    print("Copy the text block below and paste it into your requirements.txt file.")
    print("-" * 27)
    # Print the final, clean list
    for req in sorted(requirements_list):
        print(req)
    print("-" * 27)


# # --- Run the function ---
# capture_key_dependency_versions()
