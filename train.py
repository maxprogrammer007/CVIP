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
from utils.visualize import plot_training_curves

def create_dummy_data(num_samples=20, img_size=224):
    """Generates dummy image tensors and labels for testing the pipeline."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(images, labels)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Train Robust AI Detector')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--use_defense', action='store_true', help='Use XAI-guided defense')
    parser.add_argument('--lambda_consist', type=float, default=0.5, help='Weight for L1 consistency loss')
    parser.add_argument('--lambda_suppress', type=float, default=0.5, help='Weight for vulnerability suppression loss')
    parser.add_argument('--lambda_contrast', type=float, default=0.1, help='Weight for feature contrastive alignment')
    parser.add_argument('--ai_weight', type=float, default=1.2, help='Cross-entropy weight for AI class to penalize false positives')
    parser.add_argument('--dry_run', action='store_true', help='Use dummy data to test pipeline')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face Token for streaming dataset')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='Limit batches per epoch (None = use full dataset)')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'vit_b_16'], help='Backbone model')
    parser.add_argument('--save_path', type=str, default='outputs/model_best.pth', help='Path to save the trained model')
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
        print('Initializing local dataset (AI-Generated vs Real)...')
        # Setting up transforms
        train_transform = get_transforms(img_size=224, is_train=True)
        val_transform = get_transforms(img_size=224, is_train=False)
        
        # Initializing datasets
        train_dataset = AIDetectionDataset(
            root_dir="data/root_dataset", 
            split="train", 
            transform=train_transform
        )
        val_dataset = AIDetectionDataset(
            root_dir="data/root_dataset", 
            split="val", 
            transform=val_transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Setup Model, Attacker, Explainer, and Loss
    model = AIDetector(model_name=args.model_name, pretrained=not args.dry_run).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    attacker = AdversarialAttacker(model, attack_type='PGD', eps=8/255, alpha=2/255, steps=2)
    explainer = XAIExplainer(model, method='IntegratedGradients')
    
    # Apply class weights to combat AI-detection false negative bias
    class_weights = torch.tensor([1.0, args.ai_weight]).to(device)
    
    loss_fn = ExplanationConsistencyLoss(
        lambda_consist=args.lambda_consist, 
        lambda_suppress=args.lambda_suppress,
        lambda_contrast=args.lambda_contrast,
        class_weights=class_weights
    )
    
    # 3. Setup Trainer
    trainer = RobustTrainer(model, optimizer, loss_fn, explainer, attacker, device=device)
    
    # Track metrics for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_clean_acc': [],
        'val_adv_acc': []
    }
    
    # 4. Training Loop
    best_clean_acc = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        train_loss, train_acc, train_stab = trainer.train_epoch(
            train_loader,
            use_defense=args.use_defense,
            lambda_consist=args.lambda_consist,
            lambda_suppress=args.lambda_suppress,
            lambda_contrast=args.lambda_contrast,
            steps_per_epoch=args.steps_per_epoch
        )
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Exp Stability: {train_stab:.4f}')
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

    # 5. Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f'\nModel saved to {args.save_path}')

    # 6. Final Evaluation on val set
    print('\n--- Final Evaluation ---')
    clean_acc, adv_acc = evaluate_model(
        model,
        val_loader,
        attacker,
        explainer,
        device=device,
        max_steps=None
    )
    print(f'Validation Clean Acc: {clean_acc:.4f} | Adv Acc: {adv_acc:.4f}')

    # 7. Plot training curves
    if len(history['train_loss']) > 0:
        plot_training_curves(
            history['train_loss'], history['train_acc'],
            [], [],
            save_path='outputs/training_curves.png'
        )

    print('Training Complete.')

if __name__ == '__main__':
    main()
