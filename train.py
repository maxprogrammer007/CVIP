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
        print('Real dataset loading would go here using AIDetectionDataset.')
        return
    
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
        train_loss, train_acc = trainer.train_epoch(train_loader, use_defense=args.use_defense)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        
    # 5. Final Evaluation
    print('\\n--- Final Evaluation ---')
    clean_acc, adv_acc = evaluate_model(model, val_loader, attacker, explainer, device=device)
    print(f'Validation Clean Acc: {clean_acc:.4f} | Adv Acc: {adv_acc:.4f}')
    
    print('Training Complete.')

if __name__ == '__main__':
    main()
