import os
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

class AIDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Directory with the data. Expected to have 'train', 'val', 'test' subdirs.
                            Inside each split, there should be 'human' and 'ai' subdirectories.
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        self.samples = []
        self.labels = []
        
        # 0 for human, 1 for AI
        class_to_idx = {'human': 0, 'ai': 1}
        
        # We handle case where user might not have set directories yet
        if os.path.exists(self.root_dir):
            for class_name, label in class_to_idx.items():
                class_dir = os.path.join(self.root_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            self.samples.append(os.path.join(class_dir, img_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return None, None
            
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # return a blank image if loading fails
            image = Image.new("RGB", (224, 224))
            
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, label

class HFStreamingDataset(IterableDataset):
    def __init__(self, dataset_name="Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", transform=None, token=None):
        """
        Streaming dataset for Hugging Face datasets.
        Args:
            dataset_name (str): Hugging Face dataset identifier.
            split (str): e.g., 'train', 'validation', 'test'.
            transform (callable, optional): Preprocessing.
            token (str, optional): Hugging Face authentication token.
        """
        # Fallback to test if user inputs val instead of validation and dataset expects validation
        if split == 'val':
            try:
                self.dataset = load_dataset(dataset_name, split='validation', streaming=True, token=token)
            except:
                self.dataset = load_dataset(dataset_name, split='val', streaming=True, token=token)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True, token=token)
            
        self.transform = transform

    def __iter__(self):
        for item in self.dataset:
            # Hugging Face image datasets typically yield PIL images in the 'image' feature
            try:
                # If the image format isn't RGB, convert it.
                if 'image' in item:
                    image = item['image'].convert('RGB')
                else:
                    # In some datasets, the image column might be named differently, but we'll assume 'image'
                    image = Image.new("RGB", (224, 224))
                    
                # The label column could be 'label' or 'target'
                if 'target' in item:
                    label = item['target']
                elif 'label' in item:
                    label = item['label']
                else:
                    label = 0 # Default if missing
                    
                # Ensure label is integer mapping (0 or 1 typically)
                label = int(label)
                
                if self.transform:
                    image = self.transform(image)
                    
                yield image, label
                
            except Exception as e:
                # Skip broken images silently to not crash the training loop
                continue
