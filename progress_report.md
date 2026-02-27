# Comprehensive Project Report: Adversarial Defense for AI Art Detection


**Objective:** To develop a robust detector capable of distinguishing Human art from AI-generated art (Stable Diffusion, Midjourney, Flux) while resisting adversarial evasion (PGD attacks).

---

## 1. Dataset Methodology
We are utilizing the **"Hemg/AI-Generated-vs-Real-Images-Datasets"** from Hugging Face.

*   **Initial Discovery**: Our early analysis found the dataset had a "Human bias" where nearly all baseline models failed to generalize human artistic structure.
*   **Data Engineering**: We implemented a custom **Stratified Sampling** pipeline to ensure balanced training:
    *   **Training Set**: 1,200 images (600 Human / 600 AI)
    *   **Validation/Test Set**: 200 images (100 Human / 100 AI)
*   **Augmentation Strategy**: We developed an **Anatomical Focus** transform using `RandomResizedCrop` and `Style-Mix` patch swapping to force the model to look at local textures rather than global statistics.

---

## 2. Technical Roadmap & Evolution

### Stage 1: The Baseline Failure
*   **Architecture**: Standard ResNet50.
*   **Outcome**: **45.0% Accuracy**.
*   **Critical Issue**: The model had **0% Human Recall**. It predicted almost everything as "AI," effectively making it a random guesser with a bias toward the penalty-heavy class.

### Stage 2: Fine-Tuning & XAI-Guided Supervision
*   **Innovation**: Introduced **XAI-Guided Attention Masking**. We used Integrated Gradients to identify "fragile" pixel regions and masked them during training.
*   **Optimization**: Reduced class weights (`ai_weight=1.5`) and lowered learning rate to `1e-5`.
*   **Outcome**: **61.0% Accuracy**. Human recall jumped to **40%**, proving the model was starting to learn legitimate artistic features.

### Stage 3: The Hardening Stage 
*   **Innovation**: 
    1.  **50/50 Adversarial Training**: Training on a mix of clean and PGD-perturbed images.
    2.  **Contrastive Alignment**: Feature-level alignment between clean and noisy versions.
    3.  **Style-Mix**: Patch-swapping to detect localized generative artifacts.
*   **Outcome**: **77.5% Accuracy**. 
*   **Key Achievement**: Balanced recall reached **~75-80%** for both classes for the first time.

---

## 3. Experimental Results Summary

| Metric | Phase 1 (Baseline) | Phase 2 (Fine-tuned) | Phase 3 (Hardened) |
| :--- | :--- | :--- | :--- |
| **Clean Accuracy** | 45.0% | 61.0% | **77.5%** |
| **Human Recall** | 0.0% | 40.0% | **75.0%** |
| **AI Recall** | 90.0% | 82.0% | **80.0%** |
| **AUC-ROC** | 0.520 | 0.721 | **0.833** |
| **Adversarial (PGD)** | <5% | 18.5% | 10.5% |

---

## 4. Current Engineering Focus
We are currently addressing the "Adversarial Flip" seen in Phase 3 results.

1.  **Metric Stability**: Implementing **IG Smoothness** (SmoothGrad) to stabilize the defense masks.
2.  **Geometric Separation**: Replacing standard MSE with **Triplet Margin Loss (Margin=1.5)** to push Human and AI feature clusters further apart.
3.  **Logit Calibration**: Implementing **Logit Squeezing** to reduce overconfidence and smooth the decision surface.

