import os
import torch
from torchvision import transforms
from PIL import Image

def get_transforms(img_size=224, is_train=True):
    """
    Returns preprocessing transformations.
    Args:
        img_size (int): Size to resize images to.
        is_train (bool): If True, applies data augmentation.
    Returns:
        torchvision.transforms.Compose
    """
    
    # Standard ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), # Loosened anatomical crops
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)) # Random occlusion
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def denormalize(tensor):
    """
    Denormalizes a tensor image for visualization.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Clone to avoid modifying the original tensor
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)

def apply_style_mix(images, labels, patch_size=32):
    """
    Implementation of Style-Mix: Swaps random patches between images in a batch.
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        labels (torch.Tensor): Batch of labels (B,).
        patch_size (int): Size of the square patch to swap.
    Returns:
        torch.Tensor: Mixed images.
    """
    B, C, H, W = images.shape
    if B < 2:
        return images
        
    mixed_images = images.clone()
    
    # Find one AI and one Human sample to swap between if possible
    ai_idx = (labels == 1).nonzero(as_tuple=True)[0]
    human_idx = (labels == 0).nonzero(as_tuple=True)[0]
    
    if len(ai_idx) > 0 and len(human_idx) > 0:
        # Swap patch from random human image to random AI image
        h_i = human_idx[torch.randint(0, len(human_idx), (1,))]
        a_i = ai_idx[torch.randint(0, len(ai_idx), (1,))]
        
        y = torch.randint(0, H - patch_size, (1,))
        x = torch.randint(0, W - patch_size, (1,))
        
        # Swap patches
        mixed_images[a_i, :, y:y+patch_size, x:x+patch_size] = images[h_i, :, y:y+patch_size, x:x+patch_size]
        mixed_images[h_i, :, y:y+patch_size, x:x+patch_size] = images[a_i, :, y:y+patch_size, x:x+patch_size]
        
    return mixed_images
