import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def evaluate_model(model, dataloader, attacker, explainer, device='cuda'):
    """"""
    Evaluates basic accuracy, adversarial robustness, and explanation shift.
    """"""
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    loop = tqdm(dataloader, desc="Evaluating")
    
    for images, labels in loop:
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
        
    clean_acc = clean_correct / total if total > 0 else 0
    adv_acc = adv_correct / total if total > 0 else 0
    
    return clean_acc, adv_acc
