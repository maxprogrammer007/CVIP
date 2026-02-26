import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from utils.visualize import plot_xai_comparison, plot_confusion_matrix, plot_roc_curve


def evaluate_model(model, dataloader, attacker, explainer, device='cuda', max_steps=None, save_dir="outputs"):
    """
    Evaluates the model across a comprehensive suite of metrics:
    - Clean Accuracy
    - Adversarial Accuracy (under PGD attack)
    - Precision, Recall, F1-Score (per-class and macro)
    - AUC-ROC
    - Confusion Matrix
    - Explanation Shift (L2 distance between clean and adversarial attribution maps)
    Saves XAI visualizations, confusion matrix, and ROC curve plots.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_labels = []
    all_clean_preds = []
    all_adv_preds = []
    all_clean_probs = []   # For AUC-ROC
    all_adv_probs = []
    explanation_shifts = []

    saved_visualization = False

    loop = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loop):
            if max_steps is not None and i >= max_steps:
                break

            images, labels = images.to(device), labels.to(device)

            # --- Clean Inference ---
            clean_logits = model(images)
            clean_probs = torch.softmax(clean_logits, dim=1)
            clean_preds = torch.argmax(clean_logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_clean_preds.extend(clean_preds.cpu().numpy())
            all_clean_probs.extend(clean_probs[:, 1].cpu().numpy())  # Prob of class 1 (AI)

            # --- Adversarial Inference ---
            if attacker is not None:
                torch.set_grad_enabled(True)
                adv_images = attacker.generate(images, labels).detach()
                torch.set_grad_enabled(False)

                adv_logits = model(adv_images)
                adv_probs = torch.softmax(adv_logits, dim=1)
                adv_preds = torch.argmax(adv_logits, dim=1)

                all_adv_preds.extend(adv_preds.cpu().numpy())
                all_adv_probs.extend(adv_probs[:, 1].cpu().numpy())

            # --- XAI Explanation Shift ---
            if explainer is not None and attacker is not None:
                torch.set_grad_enabled(True)
                images_grad = images.clone().detach().requires_grad_(True)
                adv_images_grad = adv_images.clone().detach().requires_grad_(True)

                clean_attrs = explainer.generate_explanation(images_grad, labels)
                adv_attrs = explainer.generate_explanation(adv_images_grad, labels)

                # Per-sample L2 distance between explanation maps
                shift = (clean_attrs - adv_attrs).norm(p=2, dim=(1, 2, 3))
                explanation_shifts.extend(shift.detach().cpu().numpy())

                torch.set_grad_enabled(False)

                # Save XAI comparison visualization for first batch
                if not saved_visualization:
                    plot_xai_comparison(
                        clean_imgs=images.detach(),
                        adv_imgs=adv_images.detach(),
                        clean_attrs=clean_attrs.detach(),
                        adv_attrs=adv_attrs.detach(),
                        labels=labels,
                        preds=clean_preds,
                        adv_preds=adv_preds,
                        save_path=os.path.join(save_dir, "xai_comparison.png")
                    )
                    saved_visualization = True

    # --- Compute Metrics ---
    all_labels = np.array(all_labels)
    all_clean_preds = np.array(all_clean_preds)
    all_adv_preds = np.array(all_adv_preds) if all_adv_preds else None
    all_clean_probs = np.array(all_clean_probs)
    all_adv_probs = np.array(all_adv_probs) if all_adv_probs else None

    total = len(all_labels)

    clean_acc = (all_clean_preds == all_labels).sum() / total
    adv_acc = (all_adv_preds == all_labels).sum() / total if all_adv_preds is not None else 0.0

    class_names = ["Human", "AI"]

    # Classification report (includes per-class P, R, F1)
    print("\n--- Clean Classification Report ---")
    print(classification_report(all_labels, all_clean_preds, target_names=class_names, zero_division=0))

    if all_adv_preds is not None:
        print("--- Adversarial Classification Report ---")
        print(classification_report(all_labels, all_adv_preds, target_names=class_names, zero_division=0))

    # AUC-ROC
    clean_auc = roc_auc_score(all_labels, all_clean_probs) if len(np.unique(all_labels)) > 1 else float('nan')
    adv_auc = roc_auc_score(all_labels, all_adv_probs) if (all_adv_probs is not None and len(np.unique(all_labels)) > 1) else float('nan')

    print(f"\nClean AUC-ROC:        {clean_auc:.4f}")
    print(f"Adversarial AUC-ROC:  {adv_auc:.4f}")

    # Macro metrics
    clean_f1     = f1_score(all_labels, all_clean_preds, average='macro', zero_division=0)
    clean_prec   = precision_score(all_labels, all_clean_preds, average='macro', zero_division=0)
    clean_recall = recall_score(all_labels, all_clean_preds, average='macro', zero_division=0)

    print(f"\nClean Macro F1:        {clean_f1:.4f}")
    print(f"Clean Macro Precision: {clean_prec:.4f}")
    print(f"Clean Macro Recall:    {clean_recall:.4f}")

    if all_adv_preds is not None:
        adv_f1     = f1_score(all_labels, all_adv_preds, average='macro', zero_division=0)
        adv_prec   = precision_score(all_labels, all_adv_preds, average='macro', zero_division=0)
        adv_recall = recall_score(all_labels, all_adv_preds, average='macro', zero_division=0)

        print(f"\nAdv Macro F1:          {adv_f1:.4f}")
        print(f"Adv Macro Precision:   {adv_prec:.4f}")
        print(f"Adv Macro Recall:      {adv_recall:.4f}")

    # Explanation shift
    if explanation_shifts:
        mean_shift = np.mean(explanation_shifts)
        std_shift  = np.std(explanation_shifts)
        print(f"\nMean XAI Shift (L2):  {mean_shift:.4f} ± {std_shift:.4f}")

    # --- Save Plots ---
    cm_clean = confusion_matrix(all_labels, all_clean_preds)
    plot_confusion_matrix(cm_clean, class_names, title="Clean Confusion Matrix",
                          save_path=os.path.join(save_dir, "cm_clean.png"))

    if all_adv_preds is not None:
        cm_adv = confusion_matrix(all_labels, all_adv_preds)
        plot_confusion_matrix(cm_adv, class_names, title="Adversarial Confusion Matrix",
                              save_path=os.path.join(save_dir, "cm_adversarial.png"))

    plot_roc_curve(all_labels, all_clean_probs, all_adv_probs,
                   save_path=os.path.join(save_dir, "roc_curve.png"))

    return clean_acc, adv_acc
