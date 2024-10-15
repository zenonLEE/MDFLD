# Multi-Domain Face Landmark Detection with Diffusion Model

This repository contains the code and dataset for our ICASSP 2024 paper titled "Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model".

## Abstract
Recently, deep learning-based facial landmark detection for in-the-wild faces has achieved significant improvement. However, there are still challenges in face landmark detection in other domains (e.g., cartoon, caricature, etc.) due to the scarcity of annotated training data. We propose a two-stage training approach leveraging limited datasets and a pre-trained diffusion model to generate synthetic data pairs in multiple domains...

## Proposed Method
Our method is based on a two-stage training framework using a latent diffusion model. First, we generate face images conditioned on facial landmarks. Then, we fine-tune the model on a small multi-domain dataset with text prompts...

## Dataset
We created a large multi-domain facial landmark dataset with 25 different styles, each containing 400 images and annotations. The dataset is publicly available in this repository.
