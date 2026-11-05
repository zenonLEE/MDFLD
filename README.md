<div align="center">

# Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model

**MDFLD** · Cartoon face landmark detection · ICASSP 2024

Yuanming Li · Gwantae Kim · Jeong-gi Kwak · Bon-hwa Ku · Hanseok Ko<br>
Korea University

[![arXiv](https://img.shields.io/badge/arXiv-2401.13191-b31b1b.svg)](https://arxiv.org/abs/2401.13191)
[![ICASSP 2024](https://img.shields.io/badge/ICASSP-2024-1f6feb.svg)](https://doi.org/10.1109/ICASSP48485.2024.10446497)
[![Project page](https://img.shields.io/badge/Project-page-8957e5.svg)](https://zenonlee.github.io/MDFLD/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-ffd21e.svg)](https://huggingface.co/zenonlee/MDFLD)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-3776ab.svg)](#quick-start)
[![Release](https://img.shields.io/github/v/release/zenonLEE/MDFLD?color=2ea44f)](https://github.com/zenonLEE/MDFLD/releases)
[![Stars](https://img.shields.io/github/stars/zenonLEE/MDFLD?style=social)](https://github.com/zenonLEE/MDFLD)

**[Use MDFLD](#quick-start)** · **[Paper](#paper-and-later-releases)** · **[Model v2 results](#results)** · **[Eyelid curves](#eyelid-contours-v21)** · **[Dataset](#synthetic-dataset-icassp-2024)** · **[Citation](#citation)**

<img src="docs/assets/teaser.jpg" width="100%" alt="68 landmarks and eyelid curves on a photo, three paintings and four frames of 3D animation">

<sub>Output of <code>mdfld image</code> with default settings; under each face, a zoom of the eyes. None of these images was used for training. <a href="#acknowledgements">Image credits</a>.</sub>

</div>

MDFLD is a 68-point face landmark detector for cartoons, caricatures, paintings and photos, trained with synthetic faces from a diffusion model. It comes as a pip package for images and videos.

## Paper and later releases

The **ICASSP 2024 paper** introduces a two-stage, landmark-conditioned diffusion approach for generating paired face images and landmarks across domains, then uses the synthetic data to fine-tune a STAR landmark detector. The released synthetic dataset covers **25 styles × 400 images (10,000 image–landmark pairs)**.

**[Published paper / DOI](https://doi.org/10.1109/ICASSP48485.2024.10446497)** · **[Open-access preprint](https://arxiv.org/abs/2401.13191)** · **[Paper results, Table 1](https://arxiv.org/html/2401.13191v1#S4.T1)** · **[Download BibTeX](https://zenonlee.github.io/MDFLD/citation.bib)**

The **2026 releases** add the cartoon model v2, SAM 3.1 eyelid contours and the image/video Python package. These additions were not part of the 2024 paper. The v2 results below use a different evaluation protocol and should not be cited as the paper's original results.

## News

- **2026-11 · pip package** for images and videos. See [Quick start](#quick-start).
- **2026-11 · Eyelid contours (v2.1)**: smooth eyelid curves without the lash band. See [Eyelid contours](#eyelid-contours-v21).
- **2026-11 · Cartoon model v2**: more accurate than the paper model. See [Cartoon model v2](#cartoon-model-v2).
- **2024 · ICASSP 2024**: paper and synthetic dataset. See [Synthetic dataset](#synthetic-dataset-icassp-2024).

## Quick start

Python 3.10–3.12 with a CUDA GPU; install PyTorch 2.4 or newer first.

For automatic face finding and eyelid curves, request access to the gated [SAM 3.1 weights](https://huggingface.co/facebook/sam3.1), then authenticate after installation:

```bash
pip install "mdfld[sam] @ git+https://github.com/zenonLEE/MDFLD"
hf auth login

mdfld image face.png --out face.json --vis face_mdfld.png
mdfld video clip.mp4 --out clip.json --vis clip_mdfld.mp4
```

```python
from mdfld import MDFLD

m = MDFLD()                                      # mode: "default", "fast" or "best"
r = m.image("face.png")                          # face box, 68 landmarks, eyelid curves
v = m.video("clip.mp4", vis="clip_mdfld.mp4")    # the same for every frame, smoothed over time
```

If you already have a local `sam3.1_multiplex.pt`, pass it with `--sam-checkpoint` instead of authenticating. Without a CUDA GPU, use a supplied face box and disable eyelids; see the CPU example in [Python package](#python-package).

## Python package

The package finds the face with SAM 3.1, predicts 68 landmarks with the v2 model (TorchScript, downloaded on first use) and fits eyelid curves.

| Mode | Eyelid step | Time per image | Difference from `best` |
|---|---|---|---|
| `best` | the v2.1 setup (SAM 3.1 video predictor) | ~5.0 s | – |
| `default` | SAM image API, 5 crops per eye | ~2.5 s | median 0.9 px |
| `fast` | SAM image API, 3 crops per eye | ~1.9 s | median 1.1 px |

Times for a 1404 × 1404 image on one NVIDIA RTX PRO 6000; differences on 28 cartoon eyes.

<details>
<summary>Details</summary>

- **Install.** Install PyTorch and torchvision for your CUDA version before `mdfld`, so pip does not replace them. The landmark model is cached in `~/.cache/mdfld` (or `$MDFLD_HOME`); `MDFLD_SAM3_CKPT` can point to a local SAM 3.1 checkpoint.
- **Without SAM or a GPU**, landmarks still work with a given face box: `mdfld image face.png --box 120,80,620,640 --device cpu --no-eyelids --out face.json`.
- **Output.** At most one face per image or frame; when no face is found, `face_box`, `landmarks` and `eyelids` are `null`. `landmarks`: 68 points, iBUG 300-W order, pixels of the upright image (EXIF orientation applied); arrays passed to the API are RGB. `eyelids`: `L` and `R` eyes with 60-point `upper` and `lower` curves and a `source`: `sam`, or `landmarks` when SAM found no eye or its curve disagreed with the eye landmarks (then the curves pass through the 6 eye landmarks). Image CLI JSON maps each input path to its result; video JSON contains `fps` and `frames`.
- **Landmarks** use three passes: the face box, then twice 1.1 × the previous points' box.
- **Video.** The face is tracked across frames and each shot is smoothed with a forward–backward [One Euro filter](https://gery.casiez.net/1euro/) (no lag; tune with `--min-cutoff` and `--beta`, or `--no-smooth`). `--every N` runs SAM for the eyelids every N frames and interpolates in between (`source: interpolated`). About 1.2 s per frame with `--mode fast`, 0.5 s with `--every 3`.

</details>

## Cartoon model v2

The paper model (v1) fine-tuned with self-training on cartoon and caricature faces. Download from the [v2.0 release](https://github.com/zenonLEE/MDFLD/releases/tag/v2.0): `mdfld_cartoon_v2.torchscript.pt` (used by the package; also on [Hugging Face](https://huggingface.co/zenonlee/MDFLD)) or `mdfld_cartoon_v2.pkl` (for the [original code](#paper-model-v1-and-the-original-code)).

### Results

CariFace test set (1,560 caricatures), NME in % of the outer eye-corner distance; eye failure = eye NME above 10%.

| Model | All 68 | Eyes | Mouth | Nose | Eyebrows | Jaw | Eye failures |
|---|---|---|---|---|---|---|---|
| v1 (paper) | 6.70 | 3.49 | 6.29 | 4.81 | 6.77 | 10.42 | 2.6% |
| **v2** | **5.03** | **2.78** | **4.62** | **3.80** | **5.01** | **7.75** | **0.9%** |

On 313 held-out real-face video frames (automatic labels), v2 reaches 1.74% (v1: 3.03%).

<details>
<summary>Other crops and protocol</summary>

| Crop | Model | All 68 | Eyes | Mouth | Nose | Eyebrows | Jaw | Eye failures |
|---|---|---|---|---|---|---|---|---|
| `demo.py` crop, 1 pass | v1 | 7.81 | 4.57 | 6.44 | 5.20 | 10.48 | 11.51 | 6.8% |
| | v2 | **6.43** | **3.80** | **4.88** | **3.95** | **9.95** | **9.36** | **5.1%** |
| Tight crop, 2 passes | v1 | 8.31 | 3.82 | 6.30 | 4.96 | **12.33** | 13.24 | 3.3% |
| | v2 | **7.63** | **3.17** | **4.73** | **3.87** | 13.63 | **12.67** | **1.7%** |
| Package setting, 3 passes | v1 | 6.70 | 3.49 | 6.29 | 4.81 | 6.77 | 10.42 | 2.6% |
| | v2 | **5.03** | **2.78** | **4.62** | **3.80** | **5.01** | **7.75** | **0.9%** |

- Square crops centred on the box (w × h) of the ground-truth points: `demo.py` 1.05 × min(w, h); tight 0.9 × max(w, h), second pass on the first-pass points; package 1.0 × max(w, h), then twice 1.1 × max(w, h) of the previous points. The tight crop cuts off jaw and eyebrow points, which then end up too far inside.
- The test images were not used for training, but their labels were used to tune the pseudo-labelling. No flip test; not directly comparable with the paper's tables.

</details>

### Training

<details>
<summary>Recipe and data</summary>

6,000 iterations from v1, batch 48, Adam (lr 1e-4, warm-up, cosine decay), EMA 0.998; square crops of 0.78–1.12 × the landmark box with shift, rotation, flips and colour, blur and JPEG augmentation; smooth-L1 on the coordinates plus AWing on the heatmaps. Pseudo-labels come from two rounds of self-training (flip-averaged, filtered by flip agreement and eye shape); their jaw points are not used.

| Data | Style | Faces | Labels | Share |
|---|---|---|---|---|
| [CariFace](https://github.com/Juyong/CaricatureFace) (train split) | caricatures | 5,940 | ground truth | 45.8% |
| [iCartoonFace](https://github.com/luxiangju-PersonAI/iCartoonFace) | 2D animation | 4,766 | pseudo | 10.4% |
| [anime-faces-256](https://huggingface.co/datasets/puruchinera/anime-faces-256) | anime | 7,673 | pseudo | 16.7% |
| [Cartoon Set](https://google.github.io/cartoonset/) (Google, CC BY 4.0) | flat avatars | 4,586 | pseudo | 10.4% |
| [pixar-images](https://huggingface.co/datasets/lavaman131/pixar-images) | 3D animation | 107 | pseudo | 4.2% |
| In-house video frames | real faces | 416 | automatic | 12.5% |

The architecture is STAR's `StackedHGNetV1` (68 points, 256 × 256 input), unchanged from v1. If you use v2, please also cite CariFace (Cai et al., Graphical Models 2021) and iCartoonFace (Zheng et al., ACM MM 2020).

</details>

### Limitations

- Benchmarked on caricatures and (with automatic labels) real faces only.
- Six points per eye: very round cartoon eyes get an almond shape (the eyelid curves help).
- Profile views, heavy beards and sketchy drawings are hard.

### Terms of use

Research use only, without warranty. The training data includes research-only (CariFace) and third-party cartoon material; please follow each dataset's terms.

## Eyelid contours (v2.1)

Dense upper and lower eyelid curves without the lash band, computed from the v2 landmarks with SAM 3.1. No training needed. The pip package runs this step and falls back to the landmark curves when SAM's curve disagrees with the landmarks (for example on oil paintings with dark eye whites, or eyes behind glasses).

<details>
<summary>Stand-alone script</summary>

Needs SAM 3 ([facebookresearch/sam3](https://github.com/facebookresearch/sam3), tested at `f1647e8`), `opencv-python scipy pillow`, and `sam3.1_multiplex.pt`. Landmarks: 68 points per face (iBUG order, pixels of the stored image) as a text file, `.npy` or JSON.

```bash
python eyelid/eyelid_sam31.py --sam3-ckpt sam3.1_multiplex.pt \
    --image face.png --landmarks face_68.txt --out face_eyelids.json --vis face_eyelids.png
```

Use `--batch landmarks.json --image-root images/` for many images. Each eye gets `upper` and `lower` (60 points, left to right), `landmark_corners` and `blunt_ends`; an eye is `null` when SAM finds none. About 4 s per image on one RTX PRO 6000.

</details>

## Synthetic dataset (ICASSP 2024)

The paper trains a landmark-conditioned diffusion model on real faces, adapts it to new styles with a few examples and text prompts, and uses the generated image–landmark pairs to fine-tune a landmark detector.

| Styles | Images per style | Resolution | Annotations |
|---|---|---|---|
| 25 (caricature, Pixar-style, hand-drawn, artistic portrait, …) | 400 | 256 × 256 | 68 landmarks |

**[Download from Google Drive](https://drive.google.com/file/d/1taZfY8_IETJG2DkhXxv7U3JpPEokkBb4/view?usp=sharing)**

## Paper model (v1) and the original code

<details>
<summary>STAR-based training and test code</summary>

```bash
git clone https://github.com/zenonLEE/MDFLD.git && cd MDFLD
pip install -r requirements.txt
```

Models: [`mdfld_cartoon_v2.pkl`](https://github.com/zenonLEE/MDFLD/releases/download/v2.0/mdfld_cartoon_v2.pkl) (SHA-256 `f061f616…9d65e`), or the paper model v1, [`best_model.zip`](https://drive.google.com/file/d/1OjZDqkt41HAqAE5mJ586KMj8BUEHC6zn/view?usp=drive_link) (place `best_model.pkl` at `./pretrained_model/cartoon_dataset_400_v2/starv2_smoothl1/model/best_model.pkl`, the default).

```bash
# with annotations (<annot_dir>/<data_definition>/test.tsv)
python test.py --mode=test --device_ids=0 --image_dir=/path/to/images --annot_dir=/path/to/annotations \
               --data_definition=CariFace_testset --pretrained_weight=/path/to/model.pkl
# images only
python test.py --mode=test_image --device_ids=0 --image_dir=/path/to/images --pretrained_weight=/path/to/model.pkl
```

For 68-point models, use any `--data_definition` except `WFLW` and `COFW`, which build 98- and 29-point networks.

</details>

## Citation

```bibtex
@inproceedings{li2024towards,
  title={Towards Multi-Domain Face Landmark Detection with Synthetic Data from Diffusion Model},
  author={Li, Yuanming and Kim, Gwantae and Kwak, Jeong-gi and Ku, Bon-hwa and Ko, Hanseok},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6730--6734},
  year={2024},
  doi={10.1109/ICASSP48485.2024.10446497},
  organization={IEEE}
}
```

## Acknowledgements

The network follows [STAR](https://github.com/ZhenglinZhou/STAR); eyelids and face finding use [SAM 3.1](https://github.com/facebookresearch/sam3) (SAM License). Figure images: public-domain works from Wikimedia Commons ([Mae Jemison](https://commons.wikimedia.org/wiki/File:Mae_Carol_Jemison.jpg), [Mona Lisa](https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg), [van Gogh](https://commons.wikimedia.org/wiki/File:Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_(454045).jpg), [Vermeer](https://commons.wikimedia.org/wiki/File:Johannes_Vermeer_(1632-1675)_-_The_Girl_With_The_Pearl_Earring_(1665).jpg)) and frames of *Sintel* (© Blender Foundation | [durian.blender.org](https://durian.blender.org), CC BY 3.0).
