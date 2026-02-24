import torch
from tqdm import tqdm

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

    def train_epoch(self, dataloader, use_defense=True, lambda_consist=0.1, lambda_suppress=0.0, steps_per_epoch=None):
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
        batches = 0
        
        self.loss_fn.lambda_consist = lambda_consist if use_defense else 0.0
        self.loss_fn.lambda_suppress = lambda_suppress if use_defense else 0.0
        
        loop = tqdm(dataloader, leave=False, desc="Training")
        for i, (images, labels) in enumerate(loop):
            if steps_per_epoch is not None and i >= steps_per_epoch:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Step 1: Optional adversarial perturbation
            if self.attacker is not None:
                # We do this in eval mode for attack generation (typical for PGD)
                self.model.eval()
                pert_images = self.attacker.generate(images, labels).detach()
                self.model.train()
            else:
                pert_images = images
                
            clean_explanations, pert_explanations = None, None
            
            # Step 2: XAI explanations
            if use_defense:
                # Enable gradients on inputs for explainer
                images.requires_grad = True
                pert_images.requires_grad = True
                
                clean_explanations = self.explainer.generate_explanation(images, labels)
                pert_explanations = self.explainer.generate_explanation(pert_images, labels)
                
            # Step 3: Forward pass and loss calculation
            self.model.train() # Make sure we are back in train mode
            self.optimizer.zero_grad()
            
            logits = self.model(pert_images)
            
            loss, cls_loss, reg_loss, supp_loss = self.loss_fn(logits, labels, clean_explanations, pert_explanations)
            
            # Step 4: Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            total_acc += acc.item()
            batches += 1
            
            # Extract values dynamically 
            v_reg = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            v_supp = supp_loss.item() if isinstance(supp_loss, torch.Tensor) else supp_loss
            
            loop.set_postfix({'loss': loss.item(), 'acc': acc.item(), 'reg': v_reg, 'supp': v_supp})
            
        return total_loss / max(1, batches), total_acc / max(1, batches)
