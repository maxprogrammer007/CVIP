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
