import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils.visualize import plot_xai_comparison

def evaluate_model(model, dataloader, attacker, explainer, device='cuda', max_steps=None, save_dir="outputs"):
    """
    Evaluates basic accuracy, adversarial robustness, and explanation shift.
    Saves a visualization of the first evaluated batch.
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    os.makedirs(save_dir, exist_ok=True)
    saved_visualization = False
    
    loop = tqdm(dataloader, desc="Evaluating")
    
    for i, (images, labels) in enumerate(loop):
        if max_steps is not None and i >= max_steps:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = torch.argmax(clean_logits, dim=1)
            clean_correct += (clean_preds == labels).sum().item()
            
        # Attack evaluation
        if attacker is not None:
            # Attacks require gradients, so we enable them temporarily
            torch.set_grad_enabled(True)
            adv_images = attacker.generate(images, labels).detach()
            torch.set_grad_enabled(False)
            
            adv_logits = model(adv_images)
            adv_preds = torch.argmax(adv_logits, dim=1)
            adv_correct += (adv_preds == labels).sum().item()
            
        total += labels.size(0)
        
        # In a full implementation, we'd also calculate explanation fidelity shift 
        # (e.g. by comparing intersection over union of explanation maps with ground truth masks, 
        # or measuring sparsity/concentration of the explanations).
        
        # Save XAI Visualization for the first batch
        if not saved_visualization and explainer is not None and attacker is not None:
            # Enable grads to get explanations
            images.requires_grad = True
            adv_images.requires_grad = True
            
            clean_attrs = explainer.generate_explanation(images, labels)
            adv_attrs = explainer.generate_explanation(adv_images, labels)
            
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
            
            # Disable grad tracking again
            images.requires_grad = False
            adv_images.requires_grad = False
        
    clean_acc = clean_correct / total if total > 0 else 0
    adv_acc = adv_correct / total if total > 0 else 0
    
    return clean_acc, adv_acc
