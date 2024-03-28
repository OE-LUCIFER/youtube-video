# Pretraining Large Language Model (LLM) from Scratch

This folder contains code for pretraining a Large Language Model (LLM) from scratch.

## Overview

This is code to pretrains a  LLM from scratch using a opensource dataset and opensource tokenizer. It utilizes the Transformers library for model training and Hugging Face's datasets library for data loading. The Mistral model architecture is configured with specific parameters suitable for training on free Colab environments.

## Configuration

The Mistral model configuration is customized for a smaller model with approximately 200 million parameters to work efficiently on free Colab environments. The key configuration parameters include:

- `vocab_size`: 32000
- `hidden_size`: 1024
- `intermediate_size`: 3584
- `num_hidden_layers`: 12
- `num_attention_heads`: 32
- `num_key_value_heads`: 8
- `hidden_act`: "silu"
- `max_position_embeddings`: 4096
- `pad_token_id`: 2
- `bos_token_id`: 1
- `eos_token_id`: 2

## Dataset

The script loads the training dataset from the HuggingFace's dataset here I am using [cosmopedia-20k](HuggingFaceTB/cosmopedia-20k).

## Training

The training process involves initializing the tokenizer of Opensource LLM (here i am using Mistral model's architecture and tokenizer), tokenizing the dataset, and then training the model using the SFTTrainer from the trl library. Training arguments such as batch size, learning rate, and max steps are configured within the script.

## Output

The trained model and tokenizer are saved in the `M-final` directory.
