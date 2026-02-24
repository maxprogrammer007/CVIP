import torch
from captum.attr import IntegratedGradients, Saliency, LayerGradCam

class XAIExplainer:
    def __init__(self, model, method='IntegratedGradients', target_layer=None):
        """
        XAI Explainer using Captum.
        Args:
            model: PyTorch model.
            method (str): 'IntegratedGradients', 'Saliency', or 'LayerGradCam'.
            target_layer: Needed if using 'LayerGradCam', e.g., model.backbone.layer4.
        """
        self.model = model
        self.method = method
        
        if method == 'IntegratedGradients':
            self.explainer = IntegratedGradients(self.model)
        elif method == 'Saliency':
            self.explainer = Saliency(self.model)
        elif method == 'LayerGradCam':
            if target_layer is None:
                raise ValueError("target_layer must be provided for LayerGradCam.")
            self.explainer = LayerGradCam(self.model, target_layer)
        else:
            raise ValueError(f"Unknown XAI method {method}")
            
    def generate_explanation(self, inputs, target):
        """
        Generates explanation map for the given inputs.
        Args:
            inputs: Tensor of shape (B, C, H, W)
            target: Target class index (e.g., 1 for AI-generated)
        Returns:
            attributions: Explanations having same shape as inputs (or spatial size for GradCam)
        """
        self.model.eval() # Ensure model is in eval mode for XAI
        
        if self.method in ['IntegratedGradients']:
            # Use sensible baseline (e.g. zeros) and limited steps for speed in training
            attributions = self.explainer.attribute(inputs, target=target, n_steps=10)
        elif self.method == 'Saliency':
            attributions = self.explainer.attribute(inputs, target=target)
        elif self.method == 'LayerGradCam':
            attributions = self.explainer.attribute(inputs, target=target)
            
        return attributions
