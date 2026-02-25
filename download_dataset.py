import os
import datasets
from PIL import Image
from tqdm import tqdm

def save_split(dataset, split_name, base_dir, max_samples=1000):
    for i, item in enumerate(tqdm(dataset.take(max_samples), desc=f"Saving {split_name}")):
        img = item['image']
        label = item['label'] # Assuming 0 for real, 1 for AI, or similar
        
        # Determine class name based on label (0: AiArtData, 1: RealArt)
        class_name = "human" if label == 1 else "ai"
        
        out_dir = os.path.join(base_dir, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, f"{i}.jpg")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(out_path)

if __name__ == "__main__":
    print("Loading dataset...")
    # Using a common dataset for AI vs Real images
    # Replace with another if this specific one doesn't work.
    ds_name = "Hemg/AI-Generated-vs-Real-Images-Datasets"
    
    try:
        ds = datasets.load_dataset(ds_name, streaming=True)
        
        base_dir = "data/root_dataset"
        
        # We can simulate train/val/test splits if the dataset only has train
        if 'train' in ds:
            train_stream = ds['train']
            
            # Using datasets feature to determine label mapping if possible:
            # For streaming datasets we might have to infer it. 
            # We will just map 0 to human, 1 to AI
            print("Saving train...")
            save_split(train_stream, "train", base_dir, max_samples=2000)
            
            print("Saving val...")
            save_split(train_stream.skip(2000), "val", base_dir, max_samples=200)
            
            print("Saving test...")
            save_split(train_stream.skip(2200), "test", base_dir, max_samples=200)
            
        print("Done!")
    except Exception as e:
        print(f"Error loading dataset: {e}")

