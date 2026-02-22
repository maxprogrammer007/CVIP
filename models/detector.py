import torch
import torch.nn as nn
import torchvision.models as models

class AIDetector(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        """"""
        Binary classifier for AI vs Human art detection.
        Args:
            model_name (str): Backbone model. Default 'resnet50'.
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """"""
        super(AIDetector, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 2)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented yet.")
            
    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extracts intermediate features for potential feature-level regularization.
        """
        # For ResNet50
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        
        return features
