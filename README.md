# Multi-Domain Face Landmark Detection with Diffusion Model

This repository contains the code and dataset for our ICASSP 2024 paper titled "Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model".

## Abstract
Recently, deep learning-based facial landmark detection for in-the-wild faces has achieved significant improvement. However, there are still challenges in face landmark detection in other domains (e.g., cartoon, caricature, etc.) due to the scarcity of annotated training data. We propose a two-stage training approach leveraging limited datasets and a pre-trained diffusion model to generate synthetic data pairs in multiple domains...

## Dataset
We created a large multi-domain facial landmark dataset with 25 different styles, each containing 400 images and annotations. The dataset is publicly available in this repository.
The multi-domain facial landmark dataset is available for download:
- [Download Dataset from Google Drive][(https://drive.google.com/your-link)](https://drive.google.com/file/d/1taZfY8_IETJG2DkhXxv7U3JpPEokkBb4/view?usp=sharing)

### Citation
If you use this dataset or find our work useful, please consider citing our paper:

@inproceedings{li2024towards,
  title={Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model},
  author={Li, Yuanming and Kim, Gwantae and Kwak, Jeong-gi and Ku, Bon-hwa and Ko, Hanseok},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6730--6734},
  year={2024},
  organization={IEEE}
}
