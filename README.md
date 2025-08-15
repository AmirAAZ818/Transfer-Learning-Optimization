# Transfer Learning Optimization: Normalization Techniques and Gradient Dynamics

## Overview

This repository contains the implementation of transfer learning optimization techniques, including custom adaptation heads with various normalization methods (Batch Normalization, Layer Normalization, and Filter Response Normalization) applied to a pre-trained *MobileNetV2* model for *CIFAR-10* image classification. It includes *PyTorch* code in a Jupyter notebook format, experimental results visualizations, and a report detailing the methodology, results, and analysis. The repo also incorporates *Weights & Biases* for experiment tracking and logging.

## Purpose

This project addresses challenges in deep neural network training, such as gradient instability and slow convergence, by fine-tuning a pre-trained *MobileNetV2* model on the *CIFAR-10* dataset. It systematically evaluates the impact of normalization techniques and gradient clipping on model performance, gradient dynamics, and loss landscapes through controlled experiments.

## Usage

### Dependencies

- PyTorch 2.0+
- Torchvision
- Matplotlib
- NumPy
- Pandas
- Weights & Biases (wandb)
- torchinfo

Install dependencies via:
```
pip install torch matplotlib numpy pandas wandb tqdm torchinfo
```

### Running the Code

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Transfer-Learning-Normalization-Dynamics.git
   cd Transfer-Learning-Normalization-Dynamics
   ```

2. Set up Weights & Biases (optional but recommended for experiment tracking):
   - Sign up at [wandb.ai](https://wandb.ai) and obtain your API key.
   - Run `wandb login` and paste your API key.

3. Experiments.
   - Experiments log metrics to *wandb*; adjust `wandb_config` for your project/entity.


    > View Experiment Results on [W&B](https://wandb.ai/amiraaz/Normalization%20Techniques%20and%20Gradient%20Dynamics).