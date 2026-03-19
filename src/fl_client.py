"""
fl_client.py — Flower Federated Learning Client
PPXFL: Privacy-Preserving Explainable Federated Learning for Alzheimer's Detection

Implements Flower NumPyClient for local training with optional DP-SGD via Opacus.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict

import flwr as fl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model
from centralised_train import MRIDataset, compute_class_weights, evaluate


class FlowerClient(fl.client.NumPyClient):
    """Flower federated learning client."""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 local_epochs=5, lr=1e-4, weight_decay=1e-4,
                 class_weights=None, dp_enabled=False,
                 noise_multiplier=1.1, max_grad_norm=1.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.epsilon_spent = 0.0
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train model on local data. Called each federated round.
        
        Returns:
            parameters: Updated model weights
            num_samples: Number of training samples (for weighted FedAvg)
            metrics: Dict with training metrics
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        
        optimiser = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Optional: Integrate Opacus DP-SGD
        privacy_engine = None
        if self.dp_enabled:
            try:
                from opacus import PrivacyEngine
                privacy_engine = PrivacyEngine()
                self.model, optimiser, self.train_loader = privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimiser,
                    data_loader=self.train_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
            except ImportError:
                print("  [WARNING] Opacus not installed. Running without DP.")
                self.dp_enabled = False
        
        # Local training for E epochs
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimiser.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimiser.step()
                
                epoch_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                epoch_total += labels.size(0)
                epoch_correct += predicted.eq(labels).sum().item()
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_total
        
        # Track ε if DP is enabled
        epsilon = 0.0
        if self.dp_enabled and privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            self.epsilon_spent = epsilon
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        
        num_train_samples = len(self.train_loader.dataset)
        
        return self.get_parameters(config={}), num_train_samples, {
            "train_loss": float(avg_loss),
            "train_accuracy": float(avg_acc),
            "epsilon": float(epsilon),
        }
    
    def evaluate(self, parameters, config):
        """Evaluate global model on local test data."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        
        loss, accuracy, preds, labels, probs = evaluate(
            self.model, self.val_loader, self.criterion, self.device
        )
        
        num_val_samples = len(self.val_loader.dataset)
        
        return float(loss), num_val_samples, {
            "accuracy": float(accuracy),
        }


def create_client(client_id, data_dir, model_name='vgg19', local_epochs=5,
                  batch_size=32, lr=1e-4, dp_enabled=False,
                  noise_multiplier=1.1, max_grad_norm=1.0):
    """
    Factory function to create a FlowerClient.
    
    Args:
        client_id: Client index (1-based)
        data_dir: Path to clients/ directory
        model_name: 'vgg19' or 'resnet50'
        local_epochs: E — local epochs per round
        batch_size: Training batch size
        lr: Learning rate
        dp_enabled: Enable DP-SGD via Opacus
        noise_multiplier: σ for DP noise
        max_grad_norm: C for gradient clipping
    
    Returns:
        FlowerClient instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load client data
    client_dir = os.path.join(data_dir, f'client_{client_id}')
    images = np.load(os.path.join(client_dir, 'images.npy'))
    labels = np.load(os.path.join(client_dir, 'labels.npy'))
    
    # 90/10 train/val split for local data
    n = len(images)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = MRIDataset(images[train_idx], labels[train_idx], augment=True)
    val_dataset = MRIDataset(images[val_idx], labels[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = get_model(model_name, num_classes=3, pretrained=True)
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    
    print(f"  Client {client_id}: {n_train} train / {n_val} val samples")
    print(f"    Labels: CN={np.sum(labels==0)} MCI={np.sum(labels==1)} AD={np.sum(labels==2)}")
    
    return FlowerClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        local_epochs=local_epochs,
        lr=lr,
        class_weights=class_weights,
        dp_enabled=dp_enabled,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )


def main():
    """Start a single Flower client (for distributed mode)."""
    parser = argparse.ArgumentParser(description='PPXFL FL Client')
    parser.add_argument('--client-id', type=int, required=True)
    parser.add_argument('--model', type=str, default='vgg19', choices=['vgg19', 'resnet50'])
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dp-enabled', action='store_true')
    parser.add_argument('--dp-epsilon', type=float, default=2.0)
    parser.add_argument('--noise-multiplier', type=float, default=1.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--server-address', type=str, default='[::]:8080')
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'clients')
    
    client = create_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        model_name=args.model,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dp_enabled=args.dp_enabled,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )
    
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == '__main__':
    main()
