# Generative AI Robustness: Defense via XAI

This repository contains the end-to-end experiment pipeline for the methodology proposed in **"Robust Detection of AI-Generated Art via Explainable AI-Driven Defense Against Adversarial Attacks"**.

The framework introduces a novel defense paradigm utilizing Explainable AI (XAI) attribution maps to shift the focus of deepfake detectors away from fragile high-frequency noise and toward semantically meaningful, robust features.

## Project Structure

- `data/`
  - `dataset.py`: A custom PyTorch `AIDetectionDataset` designed to load structured clean human and AI-generated image datasets.
  - `transforms.py`: Data augmentation (RandomResizedCrop, Flip, ColorJitter) and normalization pipelines.
- `models/`
  - `detector.py`: A `ResNet50`-based binary classification model (`AIDetector`) for distinguishing between human and AI art.
- `attacks/`
  - `attacker.py`: A wrapper utilizing `torchattacks` to inject perturbations like Projected Gradient Descent (PGD) and Fast Gradient Sign Method (FGSM) dynamically.
- `xai/`
  - `explainer.py`: An attribution generator wrapping `captum` supporting Integrated Gradients, Saliency, and Layer Grad-CAM to reveal the model's focus regions.
- `defense/`
  - `losses.py`: The `ExplanationConsistencyLoss` which integrates standard Cross-Entropy with an XAI-driven regularization term.
  - `trainer.py`: A custom `RobustTrainer` intertwining adversarial image generation with XAI explanation consistency during backpropagation.
- `train.py`: The main script to configure and launch robust training.
- `eval.py`: The evaluation suite to measure clean and robust accuracy.

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies via:

```bash
pip install -r requirements.txt
```

This will install PyTorch, Torchvision, Torchattacks, Captum, and WandB.

## Quickstart & Dry Run

To verify that the system is properly configured—and that parameters, adversarial perturbations, and loss graphs compute cleanly—a dry-run mode is available. This runs the pipeline using randomly generated dummy sensor data.

```bash
python train.py --dry_run --use_defense --epochs 1
```

## Running on Real Data

To run exactly as designed with your dataset:
1.  Make sure your image data is structured like:
    ```
    root_dataset/
    ├── train/
    │   ├── human/
    │   └── ai/
    ├── val/
    │   ├── human/
    │   └── ai/
    └── test/
        ├── human/
        └── ai/
    ```
2.  Update `train.py` to point the `root_dir` of the `AIDetectionDataset` to your folder (`root_dataset`).
3.  Run the pipeline without the `--dry_run` flag.

## How it works (The Defense Mechanism)

Traditional baseline classifiers are easily fooled by bounded adversarial noise because they latch onto high-frequency residuals left by diffusion processes.

Our defense calculates attribution maps for a clean image $x$, and an adversarially perturbed image $x'$. Using a custom regularization loss `ExplanationConsistencyLoss`, it penalizes the difference between the explanation maps $E(x)$ and $E(x')$, explicitly forcing the network to make decisions based on robust, human-interpretable regions (like anatomical irregularities or contextual styles) rather than invisible, fragile static noise.
