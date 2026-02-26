import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplanationConsistencyLoss(nn.Module):
    def __init__(self, lambda_consist=0.1, lambda_suppress=0.0, lambda_contrast=0.1, 
                 lambda_triplet=0.5, lambda_squeeze=0.1, class_weights=None):
        """
        Custom loss incorporating cross-entropy, an explanation consistency regularization term, 
        and an explicit vulnerability suppression penalty.
        Args:
            lambda_consist (float): Weight for L1 explanation consistency (penalizes shift). 
            lambda_suppress (float): Weight for suppressing pixel regions heavily exploited by attacks.
            lambda_contrast (float): Weight for contrastive alignment (MSE) of feature representations.
            lambda_triplet (float): Weight for Triplet Margin Loss (cluster separation).
            lambda_squeeze (float): Weight for Logit Squeezing (smoothness).
            class_weights (torch.Tensor, optional): Optional weights for handling class imbalance.
        """
        super(ExplanationConsistencyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_consist = lambda_consist
        self.lambda_suppress = lambda_suppress
        self.lambda_contrast = lambda_contrast
        self.lambda_triplet = lambda_triplet
        self.lambda_squeeze = lambda_squeeze
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.5)

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
            
        # 3. Triplet Margin Loss (Cluster Separation)
        triplet_term = 0.0
        if self.lambda_triplet > 0 and clean_features is not None and perturbed_features is not None:
            # Anchor: Clean AI, Positive: Perturbed AI, Negative: Clean/Pert Human
            ai_indices = (targets == 1).nonzero(as_tuple=True)[0]
            human_indices = (targets == 0).nonzero(as_tuple=True)[0]
            
            if len(ai_indices) > 0 and len(human_indices) > 0:
                # We can construct triplets
                # For each AI sample, pick a random Human sample as negative
                anchors = c_feat_norm[ai_indices]
                positives = p_feat_norm[ai_indices]
                # Simple negative selection: just shuffle human features if counts match, 
                # or repeat them if they don't.
                neg_indices = human_indices[torch.randint(0, len(human_indices), (len(ai_indices),))]
                negatives = c_feat_norm[neg_indices]
                
                triplet_term = self.triplet_loss(anchors, positives, negatives)
        
        # 4. Logit Squeezing
        squeeze_loss = 0.0
        if self.lambda_squeeze > 0:
            squeeze_loss = torch.mean(logits**2)

        total_loss = cls_loss + \
                     (self.lambda_consist * reg_loss) + \
                     (self.lambda_suppress * suppress_loss) + \
                     (self.lambda_contrast * contrast_loss) + \
                     (self.lambda_triplet * triplet_term) + \
                     (self.lambda_squeeze * squeeze_loss)
                     
        return total_loss, cls_loss, reg_loss, suppress_loss, contrast_loss, triplet_term, squeeze_loss

