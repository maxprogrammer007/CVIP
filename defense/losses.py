import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplanationConsistencyLoss(nn.Module):
    def __init__(self, lambda_consist=0.1, lambda_suppress=0.0):
        """
        Custom loss incorporating cross-entropy, an explanation consistency regularization term, 
        and an explicit vulnerability suppression penalty.
        Args:
            lambda_consist (float): Weight for L1 explanation consistency (penalizes shift). 
            lambda_suppress (float): Weight for suppressing pixel regions heavily exploited by attacks.
        """
        super(ExplanationConsistencyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_consist = lambda_consist
        self.lambda_suppress = lambda_suppress
        self.l1_loss = nn.L1Loss()

    def forward(self, logits, targets, clean_explanations=None, perturbed_explanations=None):
        """
        Calculates total defense loss.
        """
        cls_loss = self.ce_loss(logits, targets)
        
        reg_loss = 0.0
        suppress_loss = 0.0
        
        if clean_explanations is not None and perturbed_explanations is not None:
            # Flatten spatial dimensions
            clean_flat = clean_explanations.view(clean_explanations.size(0), -1)
            pert_flat = perturbed_explanations.view(perturbed_explanations.size(0), -1)
            
            # Normalize explanations
            clean_norm = F.normalize(clean_flat, p=2, dim=1)
            pert_norm = F.normalize(pert_flat, p=2, dim=1)
            
            # 1. Explanation Consistency (L1 distance)
            if self.lambda_consist > 0:
                reg_loss = self.l1_loss(clean_norm, pert_norm)
                
            # 2. Vulnerability Suppression
            if self.lambda_suppress > 0:
                # Vulnerability Mask M_vuln = |E_adv - E_clean|
                vuln_mask = torch.abs(pert_norm - clean_norm)
                
                # We penalize the dot product of the adversarial explanation map and the vulnerability mask.
                # E_adv * M_vuln means penalizing the model for placing attention on pixels that 
                # changed significantly under attack.
                suppress_term = pert_norm * vuln_mask
                suppress_loss = suppress_term.sum(dim=1).mean()
            
        total_loss = cls_loss + (self.lambda_consist * reg_loss) + (self.lambda_suppress * suppress_loss)
        return total_loss, cls_loss, reg_loss, suppress_loss
