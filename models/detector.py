import torch
import torch.nn as nn
import torchvision.models as models

class AIDetector(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        """
        Binary classifier for AI vs Human art detection.
        Args:
            model_name (str): Backbone model. Default 'resnet50'.
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super(AIDetector, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 2)
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, 2)
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, 2)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented yet.")
            
    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extracts intermediate features for potential feature-level regularization.
        """
        # For ResNet50 and EfficientNet_b0 and ViT, this just provides a stub 
        # to expose the final feature embeddings before the classifier head if needed.
        if self.model_name == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
        elif self.model_name == 'efficientnet_b0':
            features = self.backbone.features(x)
        elif self.model_name == 'vit_b_16':
            # ViT processes sequence of patches, returning the CLS token feature is more complex natively, 
            # so we just return the full sequence encoding before the head as a feature map equivalent.
            # Captum attribution handles ViT natively instead of relying on this manual extraction.
            x = self.backbone._process_input(x)
            n = x.shape[0]

            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.backbone.encoder(x)
            # return spatial sequence without cls to match (B, N, C) style
            features = x[:, 1:]

        return features
