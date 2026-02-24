import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.detector import AIDetector
from attacks.attacker import AdversarialAttacker
from xai.explainer import XAIExplainer
from defense.losses import ExplanationConsistencyLoss
from defense.trainer import RobustTrainer
from eval import evaluate_model
import argparse
import os

from data.dataset import AIDetectionDataset, HFStreamingDataset
from data.transforms import get_transforms

def create_dummy_data(num_samples=20, img_size=224):
    """Generates dummy image tensors and labels for testing the pipeline."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(images, labels)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Train Robust AI Detector')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_defense', action='store_true', help='Use XAI-guided defense')
    parser.add_argument('--dry_run', action='store_true', help='Use dummy data to test pipeline')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face Token for streaming dataset')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of batches per epoch (essential for infinite streaming data)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 1. Provide Data
    if args.dry_run:
        print('Running dry-run with dummy data...')
        train_dataset = create_dummy_data(32)
        val_dataset = create_dummy_data(16)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print('Initializing Hugging Face streaming dataset (AI-Generated vs Real)...')
        # Setting up transforms
        train_transform = get_transforms(img_size=224, is_train=True)
        val_transform = get_transforms(img_size=224, is_train=False)
        
        # Initializing datasets
        train_dataset = HFStreamingDataset(
            dataset_name="Hemg/AI-Generated-vs-Real-Images-Datasets", 
            split="train", 
            transform=train_transform,
            token=args.hf_token
        )
        val_dataset = HFStreamingDataset(
            dataset_name="Hemg/AI-Generated-vs-Real-Images-Datasets", 
            split="train", 
            transform=val_transform,
            token=args.hf_token
        )
        
        # For IterableDatasets, we don't shuffle via DataLoader, we just load in sequence
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 2. Setup Model, Attacker, Explainer, and Loss
    model = AIDetector(model_name='resnet50', pretrained=False).to(device)  # False for speed in testing
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    attacker = AdversarialAttacker(model, attack_type='PGD', eps=8/255, alpha=2/255, steps=2)
    explainer = XAIExplainer(model, method='IntegratedGradients')
    loss_fn = ExplanationConsistencyLoss(lambda_reg=0.1)
    
    # 3. Setup Trainer
    trainer = RobustTrainer(model, optimizer, loss_fn, explainer, attacker, device=device)
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f'\\nEpoch {epoch+1}/{args.epochs}')
        train_loss, train_acc = trainer.train_epoch(
            train_loader, 
            use_defense=args.use_defense,
            steps_per_epoch=args.steps_per_epoch if not args.dry_run else None
        )
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        
    # 5. Final Evaluation
    print('\\n--- Final Evaluation ---')
    clean_acc, adv_acc = evaluate_model(
        model, 
        val_loader, 
        attacker, 
        explainer, 
        device=device,
        max_steps=args.steps_per_epoch if not args.dry_run else None
    )
    print(f'Validation Clean Acc: {clean_acc:.4f} | Adv Acc: {adv_acc:.4f}')
    
    print('Training Complete.')

if __name__ == '__main__':
    main()
