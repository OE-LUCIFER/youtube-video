# Idefics Fine-Tuning for Vision-Language Tasks

This repository contains a script for fine-tuning the Idefics model for vision-language tasks. The script uses the Hugging Face Transformers library and the Peft library for LoRA (Layer-wise Relevance Analysis) fine-tuning.

## Overview

The script covers the following steps:

1. **Setup**: Initial configuration and model preparation. The script sets up the device (CPU or GPU), loads the Idefics checkpoint, and configures the model for 4-bit quantization using the BitsAndBytes library.

2. **Before Fine Tuning**: The script performs inference with the original model to demonstrate its initial performance.

3. **Image Preprocessing**: The script defines functions for converting images to RGB and applying transformations such as random resized cropping and normalization.

4. **Dataset Preparation**: The script loads and transforms the dataset for training. It uses the Doodles Captions dataset and splits it into training and evaluation sets.

5. **Training**: The script sets up model training parameters and LoRA configuration, then trains the model. It uses the TrainingArguments class from the Hugging Face Transformers library to configure the training parameters and the Trainer class to train the model.

6. **After Fine Tuning**: The script performs inference with the fine-tuned model to demonstrate its performance after training.

7. **Model Saving and Pushing to HuggingFace Hub**: The script saves the model and tokenizer, then pushes them to the HuggingFace Model Hub.

## Requirements

- Python 3.7 or higher
- Hugging Face Transformers library
- Peft library
- BitsAndBytes library
- Torchvision library
- Google Colab environment (for Google Drive integration and GPU access)

