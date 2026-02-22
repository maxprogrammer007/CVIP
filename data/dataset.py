import os
from PIL import Image
from torch.utils.data import Dataset

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
