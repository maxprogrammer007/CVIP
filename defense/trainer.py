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

    def train_epoch(self, dataloader, use_defense=True, lambda_reg=0.1):
        """
        Runs one epoch of training.
        Args:
            dataloader: PyTorch train dataloader.
            use_defense: If True, uses the XAI-driven regularization on top of adversarial training.
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        self.loss_fn.lambda_reg = lambda_reg if use_defense else 0.0
        
        loop = tqdm(dataloader, leave=False, desc=\"Training\")
        for images, labels in loop:
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
            
            # Step 2: XAI explanations (requires gradients w.r.t inputs, so model is locally in eval mode
            # inside explainer. However, for clean implementation and efficiency we only do this if defense is active)
            if use_defense:
                # Enable gradients on inputs for explainer
                images.requires_grad = True
                pert_images.requires_grad = True
                
                clean_explanations = self.explainer.generate_explanation(images, labels)
                pert_explanations = self.explainer.generate_explanation(pert_images, labels)
                
            # Step 3: Forward pass and loss calculation
            self.model.train() # Make sure we are back in train mode
            self.optimizer.zero_grad()
            
            # Train on the perturbed images (Adversarial Training)
            # or clean images if no attacker is provided
            logits = self.model(pert_images)
            
            loss, cls_loss, reg_loss = self.loss_fn(logits, labels, clean_explanations, pert_explanations)
            
            
            # Step 4: Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            total_acc += acc.item()
            
            loop.set_postfix({'loss': loss.item(), 'acc': acc.item(), 'reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss})
            
        return total_loss / len(dataloader), total_acc / len(dataloader)
