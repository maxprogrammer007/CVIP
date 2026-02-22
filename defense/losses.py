import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplanationConsistencyLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        """
        Custom loss incorporating traditional cross-entropy with a regularization term 
        that enforces consistency between explanation maps of clean and perturbed images. 
        Args:
            lambda_reg (float): Weight for the explanation consistency regularization. 
        """
        super(ExplanationConsistencyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_reg = lambda_reg
        self.l1_loss = nn.L1Loss()

    def forward(self, logits, targets, clean_explanations=None, perturbed_explanations=None):
        """
        Calculates total loss.
        Args:
           logits: Model predictions (B, NumClasses).
           targets: Ground truth classes (B,).
           clean_explanations: Explanation maps for clean inputs.
           perturbed_explanations: Explanation maps for perturbed inputs.
        """
        cls_loss = self.ce_loss(logits, targets)
        
        reg_loss = 0.0
        if clean_explanations is not None and perturbed_explanations is not None:
            # We want the explanation maps to remain relatively stable under attack
            # penalizing reliance on fragile high-frequency regions.
            
            # Normalize explanations (optional but recommended for stability)
            # Flatten spatial dimensions
            clean_flat = clean_explanations.view(clean_explanations.size(0), -1)
            pert_flat = perturbed_explanations.view(perturbed_explanations.size(0), -1)
            
            clean_norm = F.normalize(clean_flat, p=2, dim=1)
            pert_norm = F.normalize(pert_flat, p=2, dim=1)
            
            # L1 distance between normalized explanations
            reg_loss = self.l1_loss(clean_norm, pert_norm)
            
        total_loss = cls_loss + self.lambda_reg * reg_loss
        return total_loss, cls_loss, reg_loss
