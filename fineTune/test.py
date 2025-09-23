# -*- coding: utf-8 -*-
# """
# testn.py
# Description
# This script runs the inference on a
# fine-tuned Gemma model using an
# image URL and a prompt.
# Author: Mazhar
# Created on: September 23, 2025
# """

# --- Unsloth must be imported before transformers, trl, peft ---
# isort: off
from unsloth import FastVisionModel
from unsloth import get_chat_template

# isort: on

import json
import os

import requests
import torch
from huggingface_hub import login
from PIL import Image


def get_hf_token():
    # Check if running in Google Colab
    try:
        import google.colab
        from google.colab import userdata

        hf_token = userdata.get("HF_TOKEN")
        print("Running in Colab: using userdata for HF_TOKEN.")
    except ImportError:
        # Not in Colab, try to load from .env file
        from dotenv import load_dotenv

        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        print("Not in Colab: using .env for HF_TOKEN.")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in Colab userdata or .env file.")
    return hf_token


hf_token = get_hf_token()
login(token=hf_token)
print(f"HF Token: {hf_token[-5:]}")

# --- Configuration ---
MODEL_NAME = "mazqoty/gemma-3n-vizWiz-finetuned"
IMAGE_URL = (
    "http://images.cocodataset.org/val2017/000000039769.jpg"  # A sample image of cats
)
# IMAGE_URL = "http://images.cocodataset.org/test-stuff2017/000000000416.jpg"

# PROMPT = "Write a short, clear description of this image."
PROMPT = """You are a helpful assistant for a visually impaired person. Your task is to describe the scene in the provided image clearly and concisely, focusing on potential obstacles or key objects."""


def run_inference(model_name, image_url, prompt):
    """
    Loads a fine-tuned model from Hugging Face, runs inference on an image, and prints the result in JSON format.
    Handles both URL and local file paths for images.
    """
    print(f"Loading model: {model_name}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,  # Use 4bit to reduce memory use and speed up inference with Unsloth.
        dtype=None,  # None for auto detection
    )
    print("Model and processor loaded successfully.")

    # Chat Template
    processor = get_chat_template(processor, "gemma-3")

    # Prepare for inference
    FastVisionModel.for_inference(model)

    try:
        print(f"Opening image: {image_url}")
        if image_url.startswith("http://") or image_url.startswith("https://"):
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        elif os.path.exists(image_url):
            image = Image.open(image_url).convert("RGB")
        else:
            print(f"Error: Image not found at {image_url}")
            return

    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate output
    print("Running inference...")
    # Set max_new_tokens to a reasonable value to avoid generating the prompt multiple times
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract only the generated text after the assistant's turn
    assistant_start = result.find("model\n")
    if assistant_start != -1:
        # Get everything after the "model" token
        full_generation = result[assistant_start + len("model\n") :]
        # Split by newline and take only the first complete line
        generated_text = full_generation.split("\n")[0]
    else:
        generated_text = "Could not extract generated text."

    # Prepare JSON-like output
    output_data = {
        "image_source": image_url,
        # "prompt": prompt,
        "generated_description": generated_text.strip(),  # Remove leading/trailing whitespace
    }

    # Print the result as JSON
    print("\n--- Inference Result (JSON) ---")
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    run_inference(MODEL_NAME, IMAGE_URL, PROMPT)
