import torchattacks

class AdversarialAttacker:
    def __init__(self, model, attack_type='PGD', eps=8/255, alpha=2/255, steps=10):
        """
        Wrapper initialized with torchattacks.
        Args:
            model: PyTorch model to be attacked.
            attack_type: 'PGD' or 'FGSM'.
            eps: Maximum perturbation.
            alpha: Step size (for PGD).
            steps: Number of iterations (for PGD).
        """
        self.model = model
        
        if attack_type == 'PGD':
            self.attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
        elif attack_type == 'FGSM':
            self.attack = torchattacks.FGSM(model, eps=eps)
        else:
            raise ValueError(f"Attack type {attack_type} not supported.")
            
        # Optional: set normalization if images are normalized before passing to model
        self.attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def generate(self, images, labels):
        """
        Generates adversarial examples.
        Args:
            images: Clean images tensor.
            labels: Ground truth labels.
        Returns:
            Perturbed images tensor.
        """
        adv_images = self.attack(images, labels)
        return adv_images
