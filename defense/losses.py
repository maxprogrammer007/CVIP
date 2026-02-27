import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplanationConsistencyLoss(nn.Module):
    def __init__(self, lambda_consist=0.1, lambda_suppress=0.0, lambda_contrast=0.1, class_weights=None):
        """
        Custom loss incorporating cross-entropy, an explanation consistency regularization term, 
        and an explicit vulnerability suppression penalty.
        Args:
            lambda_consist (float): Weight for L1 explanation consistency (penalizes shift). 
            lambda_suppress (float): Weight for suppressing pixel regions heavily exploited by attacks.
            lambda_contrast (float): Weight for contrastive alignment of feature representations.
            class_weights (torch.Tensor, optional): Optional weights for handling class imbalance.
        """
        super(ExplanationConsistencyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_consist = lambda_consist
        self.lambda_suppress = lambda_suppress
        self.lambda_contrast = lambda_contrast
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, targets, clean_explanations=None, perturbed_explanations=None, clean_features=None, perturbed_features=None):
        """
        Calculates total defense loss.
        """
        cls_loss = self.ce_loss(logits, targets)
        
        reg_loss = 0.0
        suppress_loss = 0.0
        contrast_loss = 0.0
        
        # 1. Explanation Consistency and Suppression
        if clean_explanations is not None and perturbed_explanations is not None:
            # Flatten spatial dimensions
            clean_flat = clean_explanations.view(clean_explanations.size(0), -1)
            pert_flat = perturbed_explanations.view(perturbed_explanations.size(0), -1)
            
            # Normalize explanations
            clean_norm = F.normalize(clean_flat, p=2, dim=1)
            pert_norm = F.normalize(pert_flat, p=2, dim=1)
            
            # Explanation Consistency (L1 distance)
            if self.lambda_consist > 0:
                reg_loss = self.l1_loss(clean_norm, pert_norm)
                
            # Vulnerability Suppression
            if self.lambda_suppress > 0:
                vuln_mask = torch.abs(pert_norm - clean_norm)
                suppress_term = pert_norm * vuln_mask
                suppress_loss = suppress_term.sum(dim=1).mean()
        
        # 2. Contrastive Alignment (Feature-level stability)
        if self.lambda_contrast > 0 and clean_features is not None and perturbed_features is not None:
            # Pool features if they are spatial (B, C, H, W) -> (B, C)
            if len(clean_features.shape) == 4:
                clean_feat_pooled = F.adaptive_avg_pool2d(clean_features, (1, 1)).view(clean_features.size(0), -1)
                pert_feat_pooled = F.adaptive_avg_pool2d(perturbed_features, (1, 1)).view(perturbed_features.size(0), -1)
            else:
                clean_feat_pooled = clean_features.view(clean_features.size(0), -1)
                pert_feat_pooled = perturbed_features.view(perturbed_features.size(0), -1)

            # MSE between normalized features pairs
            c_feat_norm = F.normalize(clean_feat_pooled, p=2, dim=1)
            p_feat_norm = F.normalize(pert_feat_pooled, p=2, dim=1)
            contrast_loss = self.mse_loss(c_feat_norm, p_feat_norm)
            
        total_loss = cls_loss + \
                     (self.lambda_consist * reg_loss) + \
                     (self.lambda_suppress * suppress_loss) + \
                     (self.lambda_contrast * contrast_loss)
                     
        return total_loss, cls_loss, reg_loss, suppress_loss, contrast_loss

