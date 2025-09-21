# FineTuneGemma: End-to-End Fine-Tuning Pipeline for Gemma on VizWiz

## Overview

This project provides a modular pipeline for preparing the [VizWiz dataset](https://vizwiz.cs.colorado.edu/VizWiz_final/) and fine-tuning vision-language models (such as Gemma) using Unsloth. The workflow is designed for easy use in Google Colab and supports robust configuration via a single [YAML file](fineTune/configs/configs.yaml).

---

## 1. Data Pipeline

The data pipeline is managed by the [`VizWizDataPipeline`](fineTune/data_pipeline/dataset_extraction.py) class, which automates:

- **Downloading**: Fetches VizWiz dataset components 
  
    - [annotations](https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip)
    - images, 
        - [Train](https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip)
        - [Val](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip)
        - [Test](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip)
  
  from either a subset created by using [Script](create_subset_dataset.py) and uploaded to Google Drive manually or the original VizWiz URLs by changing settings in [`configs.yaml`](fineTune/configs/configs.yaml).

- Set `use_gdrive_urls: true` in [`configs.yaml`](fineTune/configs/configs.yaml) to choose which dataset to use. Since original dataset is quite big which takes couple of hours in extraction, applying transformations and if you are only training for 1 epoch it will further take 7 or more hours. Therefore, it is highly recommended that you use subset version of dataset.

- [**Extraction**:](fineTune/data_pipeline/dataset_extraction.py) Unzips and organizes the dataset into a structured directory. It also create respective `.jsonl` files for `annotation`, `train`, `val`, and `test` splits

- [**Transformation**](fineTune/data_pipeline/dataset_transformation.py) Converts raw data into a conversational format suitable for fine-tuning vision-language models and saved to disk.
- **Integrity Checking**: Verifies that the processed data matches expectations (e.g., no data loss during transformation).

- [Load Data:](fineTune/data_pipeline/dataset_loader.py) Loads transformed data for training.

You can control each step of the pipeline via the `preparation_steps` section in [`configs.yaml`](fineTune/configs/configs.yaml).

---

## 2. Configuration: `configs.yaml`

All settings are centralized in [`fineTune/configs/configs.yaml`](fineTune/configs/configs.yaml):

- **dataset.use_gdrive_urls**:  
  Set to `true` to use Google Drive links (recommended for Colab), or `false` for original VizWiz URLs.

- **dataset.urls**:  
  Contains both original and Google Drive URLs for all dataset components.

- **dataset.paths**:  
  Specifies where data will be stored and processed.  
  - `base_dir`, `temp_zip_dir`, `images_dir`, `annotations_dir`, `processed_dir`

- **dataset.preparation_steps**:  
  Flags to enable/disable each pipeline stage:
  - `run_prepare_step`: Download and extract raw data.
  - `run_transform_step`: Transform data for tuning.
  - `check_dataset_integrity`: Verify processed data.

- **dataset.subset**:  
  Enable to use only a percentage of the data for quick experiments.

- **fineTune**:  
  - `model_to_use`: The HuggingFace model ID to fine-tune.
  - `fourbit_models`: List of supported 4-bit models.
  - Output directories for adapters, checkpoints, logs, etc.
  - `export`: Settings for saving or pushing the final model to HuggingFace Hub.

- **tensorboard**:  
  - `auto_start`: Automatically launch TensorBoard in Colab.
  - `show_instructions`: Print setup instructions for TensorBoard.

**Example:**  
See the provided [`configs.yaml`](fineTune/configs/configs.yaml) for a complete, annotated example.

---

## 3. Useful Utilities

- [`make_clean_dir`](fineTune/utils/make_clean_dir.py):  
  Safely creates or resets directories.

- [`setupTensorboard`](fineTune/utils/colab_utils.py):  
  Automatically configures TensorBoard for Colab or prints manual setup instructions for local environments.

- [`DetailedLoggingCallback`](fineTune/utils/logging_callback.py):  
  Custom callback for enhanced logging during training.

- [`display_training_summary`](fineTune/utils/training_analyzer.py):  
  Summarizes training metrics after completion.

- [`get_gpu_usage_stats`](fineTune/utils/gpu_stats.py):  
  Reports GPU memory usage before and after training.

---

## 4. Running `FineTune.ipynb` in Google Colab

### **Step-by-Step Instructions**

#### **A. Prepare Your Environment**

1. **Upload Your Code and Dataset**
   - Upload `fineTune.zip` (containing the codebase) to your Colab workspace.

2. **Mount Google Drive**
   - The notebook will prompt you to mount your Google Drive for persistent storage.

3. **Unzip and Set Up Project**
   - The notebook extracts the code and adds it to the Python path.

4. **Install Dependencies**
   - All required packages are installed automatically, including Unsloth, PyTorch, and others.

5. **Configure Hugging Face Authentication**
   - Log in using your Hugging Face token (preferably stored in Colab `userdata`).

6. **Update `configs.yaml` for Colab**
   - The notebook automatically rewrites paths in `configs.yaml` to use Colab's local disk for fast I/O and Google Drive for persistent outputs.

#### **B. Run the Data Pipeline**

- The notebook runs [`prepare_data.py`](fineTune/prepare_data.py), which:
  - Downloads and extracts the dataset.
  - Transforms it for fine-tuning.
  - Optionally checks data integrity.

#### **C. Visualize the Data**

- Use the provided visualization cell to inspect individual dataset samples before training.

#### **D. Fine-Tune the Model**

- The notebook:
  - Loads the model and processor.
  - Configures LoRA adapters.
  - Loads the processed dataset.
  - Sets up training arguments and logging.
  - Starts training with real-time TensorBoard support.

#### **E. Save and Export**

- After training, the notebook:
  - Saves LoRA adapters and processor files.
  - Optionally pushes the merged model to the Hugging Face Hub.

---

## 5. Monitoring Training

- **TensorBoard**:  
  - If `tensorboard.auto_start` is enabled and running in Colab, TensorBoard will launch automatically.
  - Otherwise, follow the printed instructions or run:
    ```
    python fineTune/monitor_training.py tensorboard
    ```
    to launch TensorBoard locally.

---

## 6. Troubleshooting

- **Missing Dependencies**:  
  Ensure all requirements are installed. The notebook will print missing packages if any are detected.

- **Path Issues**:  
  Always run the notebook from the root directory where `fineTune.zip` is extracted.

- **Data Download Errors**:  
  Check your internet connection and verify that the URLs in `configs.yaml` are accessible.

---

## 7. References

- [VizWiz Dataset](https://vizwiz.cs.colorado.edu/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Hub](https://huggingface.co/)

---

## 8. Project Structure

```
fineTune/
    configs/
        configs.yaml
    data_pipeline/
        dataset_extraction.py
        dataset_transformation.py
        ...
    utils/
        colab_utils.py
        logging_callback.py
        ...
    prepare_data.py
    train.py
    visualize_data.py
    monitor_training.py
FineTune.ipynb
```

---

**Note:**  
Files in the `extras/` folder are not required for the main workflow and can be ignored.

---