# -*- coding: utf-8 -*-
# """
# inference_cpu.py
# Description
# This script runs the inference on a
# fine-tuned Gemma model using an
# image URL and a prompt.
# Author: Mazhar
# Created on: September 23, 2025
# """

import json
import os

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import login


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


def run_cpu_inference(model_name, image_url, prompt):
    """
    Loads a fine-tuned model from Hugging Face using only the standard
    transformers library and runs inference on a CPU.
    """
    print(f"Loading model for CPU: {model_name}")

    # Use the standard AutoProcessor and AutoModelForCausalLM classes
    # This is the Hugging Face standard for loading models.
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for a smaller memory footprint
        low_cpu_mem_usage=True,  # A transformers flag to be more memory efficient on CPU
    )
    print("✅ Model and processor loaded successfully.")

    try:
        print(f"Opening image: {image_url}")
        if image_url.startswith("http://") or image_url.startswith("https://"):
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        elif os.path.exists(image_url):
            image = Image.open(image_url).convert("RGB")
        else:
            raise FileNotFoundError(f"Image not found at {image_url}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    # The chat template is part of the processor's configuration,
    # so we can apply it directly without any special functions.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]

    # The processor handles both text and images
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=image,
        return_tensors="pt",
    )
    # Note: We DO NOT move the inputs to "cuda". They will stay on the CPU.

    print("⏳ Running inference on CPU (this may take a moment)...")

    # Generate the output
    outputs = model.generate(
        **inputs, max_new_tokens=300, eos_token_id=processor.tokenizer.eos_token_id
    )

    # Decode the result
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Post-process the output to get only the assistant's response
    assistant_start = result.rfind("model\n")  # Use rfind to get the last occurrence
    if assistant_start != -1:
        generated_text = result[assistant_start + len("model\n") :]
        # Take the first line to avoid repetitions
        generated_text = generated_text.split("\n")[0].strip()
    else:
        generated_text = "Could not extract generated text."

    output_data = {
        "image_source": image_url,
        "prompt": prompt,
        "generated_description": generated_text,
    }

    print("\n--- Inference Result (JSON) ---")
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    # Ensure you have your HF_TOKEN set in the environment or Colab userdata
    hf_token = get_hf_token()
    from huggingface_hub import login

    login(token=hf_token)

    MODEL_NAME = (
        "mazhar/gemma-3-finetuned-cpu"  # Replace with your fine-tuned model name
    )
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    PROMPT = "Write a short, clear description of this image."

    run_cpu_inference(MODEL_NAME, IMAGE_URL, PROMPT)
