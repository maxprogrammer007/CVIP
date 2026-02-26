import torch
from tqdm import tqdm
from data.transforms import apply_style_mix

class RobustTrainer:
    def __init__(self, model, optimizer, loss_fn, explainer, attacker, device='cuda'):
        """
        Trainer integrating adversarial attacks and XAI-guided regularization.
        Args:
           model: The detector model.
           optimizer: PyTorch optimizer.
           loss_fn: ExplanationConsistencyLoss instance.
           explainer: XAIExplainer instance.
           attacker: AdversarialAttacker instance.
           device: Device to run on.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.explainer = explainer
        self.attacker = attacker
        self.device = device

    def train_epoch(self, dataloader, use_defense=True, lambda_consist=0.1, lambda_suppress=0.0, 
                    lambda_contrast=0.1, lambda_triplet=0.5, lambda_squeeze=0.1, steps_per_epoch=None):
        """
        Runs one epoch of training.
        Args:
            dataloader: PyTorch train dataloader.
            use_defense: If True, uses the XAI-driven regularization on top of adversarial training.
            steps_per_epoch (int): Limits number of batches due to infinite streams.
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_stability = 0.0 # Track metric: how consistent the explanations are
        batches = 0
        
        self.loss_fn.lambda_consist = lambda_consist if use_defense else 0.0
        self.loss_fn.lambda_suppress = lambda_suppress if use_defense else 0.0
        self.loss_fn.lambda_contrast = lambda_contrast if use_defense else 0.0
        self.loss_fn.lambda_triplet = lambda_triplet if use_defense else 0.0
        self.loss_fn.lambda_squeeze = lambda_squeeze if use_defense else 0.0
        
        loop = tqdm(dataloader, leave=False, desc="Training")
        for i, (images, labels) in enumerate(loop):
            if steps_per_epoch is not None and i >= steps_per_epoch:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply Style-Mix augmentation
            images = apply_style_mix(images, labels)
            
            # Step 1: Optional adversarial perturbation
            if self.attacker is not None:
                # We do this in eval mode for attack generation (typical for PGD)
                self.model.eval()
                pert_images = self.attacker.generate(images, labels).detach()
                self.model.train()
            else:
                pert_images = images
                
            clean_explanations, pert_explanations = None, None
            clean_features, pert_features = None, None
            stability = 1.0 # default
            
            # Step 2: XAI explanations and feature extraction
            if use_defense:
                # Enable gradients on inputs for explainer
                images.requires_grad = True
                pert_images.requires_grad = True
                
                # Use SmoothGrad (nt_samples=5) for stable masking
                clean_explanations = self.explainer.generate_explanation(images, labels, nt_samples=5)
                pert_explanations = self.explainer.generate_explanation(pert_images, labels, nt_samples=5)
                
                # Feature extraction for contrastive alignment
                clean_features = self.model.extract_features(images)
                pert_features = self.model.extract_features(pert_images)
                
                # Dynamic Uncertainty-Based Masking
                with torch.no_grad():
                    # Fragile regions: where shift is highest
                    vuln_mask = torch.abs(pert_explanations - clean_explanations)
                    stability = 1.0 - (vuln_mask.view(vuln_mask.size(0), -1).mean(dim=1).mean().item())
                    
                    # Normalize maps
                    mask_max = vuln_mask.view(vuln_mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
                    vuln_mask = vuln_mask / mask_max
                    
                    # Semantic Priority: Clean map high attribution regions
                    # We protect the top attribution regions of the clean map
                    clean_max = clean_explanations.view(clean_explanations.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
                    clean_norm = clean_explanations / clean_max
                    
                    # Adaptive Floor: protect regions with > 0.5 normalized clean attribution
                    adaptive_floor = (clean_norm > 0.5).float()
                    
                    # effective_mask = (1 - fragile) + semantic_important
                    effective_mask = torch.clamp((1.0 - vuln_mask) + adaptive_floor, 0.0, 1.0)
                    masked_pert_images = pert_images * effective_mask
            else:
                masked_pert_images = pert_images
                
            # Step 3: Hardened 50/50 Adversarial Training
            # We mix half clean and half masked-perturbed for the main labels
            B = images.size(0)
            if B >= 2:
                # Randomly select 50% indices to be adversarial
                adv_idx = torch.randperm(B)[:B//2]
                mixed_train_images = images.clone()
                mixed_train_images[adv_idx] = masked_pert_images[adv_idx]
            else:
                mixed_train_images = masked_pert_images
                
            # Forward pass
            self.model.train() 
            self.optimizer.zero_grad()
            
            logits = self.model(mixed_train_images)
            
            # Expanded return from updated loss_fn (Triplet, Squeeze)
            loss, cls_loss, reg_loss, supp_loss, contrast_loss, triplet_loss, squeeze_loss = self.loss_fn(
                logits, labels, 
                clean_explanations, pert_explanations,
                clean_features, pert_features
            )
            
            # Step 4: Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            total_acc += acc.item()
            total_stability += stability
            batches += 1
            
            # Extract values dynamically 
            v_trip = triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else triplet_loss
            v_sqz = squeeze_loss.item() if isinstance(squeeze_loss, torch.Tensor) else squeeze_loss
            
            loop.set_postfix({
                'loss': f"{loss.item():.3f}", 
                'acc': f"{acc.item():.3f}", 
                'stab': f"{stability:.3f}",
                'trip': f"{v_trip:.3f}",
                'sqz': f"{v_sqz:.3f}"
            })
            
        return total_loss / max(1, batches), total_acc / max(1, batches), total_stability / max(1, batches)
