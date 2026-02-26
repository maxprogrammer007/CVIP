"""
Standalone evaluation script.
Loads the saved trained model and runs the full metrics suite from eval.py.
Usage:
    python run_eval.py [--checkpoint outputs/model_best.pth]
"""
import argparse
import torch
from torch.utils.data import DataLoader

from models.detector import AIDetector
from attacks.attacker import AdversarialAttacker
from xai.explainer import XAIExplainer
from data.dataset import AIDetectionDataset
from data.transforms import get_transforms
from eval import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/model_best.pth',
                        help='Path to saved model checkpoint (.pth)')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    transform = get_transforms(img_size=224, is_train=False)
    test_dataset = AIDetectionDataset(root_dir="data/root_dataset", split="test", transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test set: {len(test_dataset)} samples")

    # Model
    model = AIDetector(model_name=args.model_name, pretrained=False).to(device)
    if args.checkpoint:
        import os
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"WARNING: checkpoint not found at {args.checkpoint}, using random weights.")

    # Attacker & Explainer
    attacker = AdversarialAttacker(model, attack_type='PGD', eps=8/255, alpha=2/255, steps=2)
    explainer = XAIExplainer(model, method='IntegratedGradients')

    # Run evaluation
    clean_acc, adv_acc = evaluate_model(
        model, test_loader, attacker, explainer,
        device=device,
        max_steps=None,
        save_dir="outputs"
    )

    print(f"\n=== Final Test Summary ===")
    print(f"Clean Accuracy:       {clean_acc:.4f}  ({clean_acc*100:.1f}%)")
    print(f"Adversarial Accuracy: {adv_acc:.4f}  ({adv_acc*100:.1f}%)")
    print(f"Robustness Drop:      {clean_acc - adv_acc:.4f} ({(clean_acc - adv_acc)*100:.1f}%)")
    print(f"\nPlots saved to:  outputs/")


if __name__ == '__main__':
    main()
