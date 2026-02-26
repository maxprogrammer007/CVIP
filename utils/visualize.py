import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Unnormalizes an image tensor for visualization.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_xai_comparison(clean_imgs, adv_imgs, clean_attrs, adv_attrs, labels, preds, adv_preds, save_path="outputs/xai_comparison.png"):
    """
    Plots a grid comparing Clean Images, Adversarial Images, and their respective XAI attribution maps.
    Also plots the absolute difference (Vulnerability Mask).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    batch_size = clean_imgs.size(0)
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
    
    if batch_size == 1:
        axes = [axes]
        
    for i in range(batch_size):
        # 1. Clean Image
        c_img = unnormalize(clean_imgs[i].clone().cpu()).permute(1, 2, 0).numpy()
        c_img = np.clip(c_img, 0, 1)
        axes[i][0].imshow(c_img)
        axes[i][0].set_title(f"Clean\\nTrue: {labels[i].item()} | Pred: {preds[i].item()}")
        axes[i][0].axis('off')
        
        # 2. Adversarial Image
        a_img = unnormalize(adv_imgs[i].clone().cpu()).permute(1, 2, 0).numpy()
        a_img = np.clip(a_img, 0, 1)
        axes[i][1].imshow(a_img)
        axes[i][1].set_title(f"Adv (PGD)\\nPred: {adv_preds[i].item()}")
        axes[i][1].axis('off')
        
        # 3. Clean Attribution
        c_attr = clean_attrs[i].cpu().numpy()
        # Average over color channels, or take max
        c_attr_heatmap = np.max(np.abs(c_attr), axis=0) 
        axes[i][2].imshow(c_attr_heatmap, cmap='hot')
        axes[i][2].set_title("Clean Attribution")
        axes[i][2].axis('off')
        
        # 4. Adversarial Attribution
        a_attr = adv_attrs[i].cpu().numpy()
        a_attr_heatmap = np.max(np.abs(a_attr), axis=0)
        axes[i][3].imshow(a_attr_heatmap, cmap='hot')
        axes[i][3].set_title("Adv Attribution")
        axes[i][3].axis('off')
        
        # 5. Vulnerability Mask (Difference)
        vuln_mask = np.abs(a_attr_heatmap - c_attr_heatmap)
        axes[i][4].imshow(vuln_mask, cmap='magma')
        axes[i][4].set_title("Vulnerability Mask\\n|Adv - Clean|")
        axes[i][4].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved XAI comparison plot to {save_path}")

def plot_training_curves(train_losses, train_accs, val_clean_accs, val_adv_accs, save_path="outputs/training_curves.png"):
    """
    Plots the training loss, training accuracy, and validation accuracies over epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Curve
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Curves
    ax2.plot(epochs, train_accs, 'g-', label='Train Acc')
    if len(val_clean_accs) > 0: # Might be empty if evaluated only at end
         ax2.plot(epochs, val_clean_accs, 'r--', label='Val Clean Acc')
         ax2.plot(epochs, val_adv_accs, 'm--', label='Val Adv Acc')
    
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves plot to {save_path}")


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path="outputs/cm.png"):
    """
    Plots a color-coded confusion matrix.
    Args:
        cm: 2D numpy array from sklearn.metrics.confusion_matrix
        class_names: list of class label strings
        title: plot title
        save_path: path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, format(cm[r, c], 'd'),
                    ha='center', va='center',
                    color='white' if cm[r, c] > thresh else 'black',
                    fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")


def plot_roc_curve(labels, clean_probs, adv_probs=None, save_path="outputs/roc_curve.png"):
    """
    Plots clean (and optionally adversarial) ROC curves with AUC annotations.
    Args:
        labels: ground truth binary labels (numpy array)
        clean_probs: model confidence for the positive class on clean images
        adv_probs: model confidence for the positive class on adversarial images (optional)
        save_path: path to save the figure
    """
    from sklearn.metrics import roc_curve, auc

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Clean ROC
    fpr_c, tpr_c, _ = roc_curve(labels, clean_probs)
    auc_c = auc(fpr_c, tpr_c)
    ax.plot(fpr_c, tpr_c, color='steelblue', lw=2, label=f"Clean (AUC = {auc_c:.3f})")

    # Adversarial ROC
    if adv_probs is not None:
        fpr_a, tpr_a, _ = roc_curve(labels, adv_probs)
        auc_a = auc(fpr_a, tpr_a)
        ax.plot(fpr_a, tpr_a, color='tomato', lw=2, linestyle='--',
                label=f"Adversarial (AUC = {auc_a:.3f})")

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve: Clean vs Adversarial', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to {save_path}")

