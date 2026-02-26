"""
Rebuilds a balanced dataset from existing cached images.
Strategy:
1. Loads the HuggingFace dataset (non-streaming) - uses local cache if available.
2. Samples n_per_class per class for train/val/test.
3. Saves into data/root_dataset/{split}/{class}/.
"""
import os, shutil, random
import datasets
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "Hemg/AI-Generated-vs-Real-Images-Datasets"
BASE_DIR = "data/root_dataset"
LABEL_MAP = {0: "ai", 1: "human"}   # 0 = AiArtData, 1 = RealArt

N_TRAIN = 600   # per class
N_VAL   = 100   # per class
N_TEST  = 100   # per class

random.seed(42)

def save_samples(samples, split, class_name):
    out_dir = os.path.join(BASE_DIR, split, class_name)
    os.makedirs(out_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(samples, desc=f"{split}/{class_name}")):
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(os.path.join(out_dir, f"{idx}.jpg"))

if __name__ == "__main__":
    print("Loading dataset (will use local cache if available)...")
    # Load in non-streaming mode - works from HF cache
    try:
        ds = datasets.load_dataset(DATASET_NAME, split='train')
    except Exception as e:
        print(f"Failed to load: {e}")
        raise

    print(f"Loaded {len(ds)} samples. Splitting by class...")

    # Group by label
    by_class = {0: [], 1: []}
    for item in tqdm(ds, desc="Grouping"):
        lbl = item['label']
        if lbl in by_class:
            by_class[lbl].append(item)

    for lbl, items in by_class.items():
        print(f"  class {lbl} ({LABEL_MAP[lbl]}): {len(items)} images")
        random.shuffle(items)

    # Clear old data
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    for lbl, class_name in LABEL_MAP.items():
        items = by_class[lbl]
        n = N_TRAIN + N_VAL + N_TEST
        if len(items) < n:
            print(f"WARNING: only {len(items)} images for class {class_name}, need {n}. Using what's available.")
            n = len(items)
        items = items[:n]
        train_items = items[:N_TRAIN]
        val_items   = items[N_TRAIN:N_TRAIN + N_VAL]
        test_items  = items[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

        save_samples(train_items, "train", class_name)
        save_samples(val_items,   "val",   class_name)
        save_samples(test_items,  "test",  class_name)

    # Verify
    for split in ["train", "val", "test"]:
        counts = {}
        for c in ["ai", "human"]:
            p = os.path.join(BASE_DIR, split, c)
            counts[c] = len(os.listdir(p)) if os.path.exists(p) else 0
        print(f"{split}: {counts}")

    print("\nDone!")
