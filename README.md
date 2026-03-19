# LoViF 2026 Semantic Image Quality Assessment (SIQA) - Final Submission

**Team:** Ayush Gupta (Participating Solo)  
**Affiliation:** Netaji Subhas University of Technology  
**CodaBench Username:** ayush13  

## Overview
This repository contains the official inference and submission generation code for our solution to the LoViF 2026 SIQA Challenge. The method utilizes a rank-consistent heterogeneous ensemble leveraging OpenCLIP, DINOv2, and various standard/neural IQA metrics to assess both perceptual fidelity and semantic plausibility.

## Prerequisites & Environment
The pipeline is designed to be highly reproducible. It dynamically downloads the required open-source pre-trained weights (DINOv2, OpenCLIP, MUSIQ, etc.) directly via PyTorch Hub and Hugging Face during execution, eliminating the need to manually download heavy `.pkl` or `.pth` files.

Ensure you have a Python environment with PyTorch installed, and then install the required dependencies:

```bash
pip install torch torchvision pyiqa open_clip_torch scikit-learn catboost xgboost lightgbm tqdm pandas scipy
```

## Directory Structure Preparation
Place the official challenge data directories (`Train`, `Val`, and `Test`) in the root directory alongside the python script.

```text
.
├── Train/               
├── Val/                 
├── Test/
├── final_code_8711.py
└── README.md
```

## Execution
To run the full end-to-end pipeline (which handles feature extraction, out-of-fold training, dynamic ensemble weight blending, and generating the exact binary-matching submission artifact):

```bash
python final_code_8711.py
```

*Note: Execution utilizes large ViT models for feature extraction. A GPU (e.g., RTX 3090/4090 class) is highly recommended. Average execution time is estimated dynamically based on hardware (~5.0s).*

## Outputs
Upon successful completion, the script will output your final package:
- `final_sub.zip`: The finalized prediction ZIP securely formatted with exact system byte-standards for CodaBench evaluation (contains `prediction.csv` and internal `readme.txt`).