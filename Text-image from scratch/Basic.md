# Diffusion Model Image Generation

This repository contains a PyTorch implementation of a diffusion model for image generation, trained from scratch on a dataset of butterfly images.

## Requirements

The following Python packages are required to run the code:

- `diffusers`
- `datasets`
- `accelerate`
- `torch`
- `torchvision`
- `PIL`
- `matplotlib`
- `tqdm`
- `numpy`
- `random`

You can install these packages using the following command:

```
%pip install diffusers datasets accelerate
```

## Usage

1. **Load and Preprocess the Dataset**: The code loads the Smithsonian Butterflies dataset from the Hugging Face Datasets library, preprocesses the images, and creates a PyTorch DataLoader.

2. **Define the U-Net Model and Noise Scheduler**: The code sets up the U-Net model and the DDPM (Denoising Diffusion Probabilistic Model) noise scheduler.

3. **Train the Model**: The code trains the diffusion model using the AdamW optimizer and the cosine annealing learning rate scheduler. It also uses the Accelerator library for mixed precision training and gradient accumulation.

4. **Generate New Images**: After training, the code uses the trained model and the DDPM pipeline to generate new images of butterflies.

## Acknowledgements

- The Smithsonian Butterflies dataset used in this project is provided by [Huggan](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset).
- The diffusion model implementation is based on the [Diffusers library](https://github.com/huggingface/diffusers) by Hugging Face.

Subscribe to my channel at https://youtube.com/@OEvortex
