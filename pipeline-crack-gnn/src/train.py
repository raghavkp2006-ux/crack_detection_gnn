"""
Config-driven training script for pipeline crack detection.

Usage:
    python -m src.train --config configs/hybrid_full.yaml
"""

import argparse
import yaml
import os
import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from typing import Optional
from sklearn.metrics import f1_score, classification_report

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc, dirichlet_energy


def load_config(path: str) -> dict:
    """Reads YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_class_weights(dataset) -> torch.Tensor:
    """Computes inverse-frequency weights for crack vs no-crack imbalance."""
    all_y = torch.cat([dataset[i].y for i in range(len(dataset))])
    num_crack = all_y.sum().item()
    num_total = all_y.numel()
    num_no_crack = num_total - num_crack

    weight_crack = num_total / (2.0 * num_crack) if num_crack > 0 else 1.0
    weight_no_crack = num_total / (2.0 * num_no_crack) if num_no_crack > 0 else 1.0

    return torch.tensor([weight_no_crack, weight_crack], dtype=torch.float32)


def get_inverted_class_weights(dataset) -> torch.Tensor:
    """Computes inverted class frequencies [n_neg/n_total, n_pos/n_total] inverted -> [n_pos/n_total, n_neg/n_total]."""
    all_y = torch.cat([dataset[i].y for i in range(len(dataset))])
    num_crack = all_y.sum().item()
    num_total = all_y.numel()
    num_no_crack = num_total - num_crack

    w_no_crack = num_crack / num_total if num_total > 0 else 0.5
    w_crack = num_no_crack / num_total if num_total > 0 else 0.5

    return torch.tensor([w_no_crack, w_crack], dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss with gamma focusing parameter and inverted class weights alpha:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = torch.nn.functional.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)

        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EarlyStopping:
    """Monitors validation F1 and saves best model checkpoint."""

    def __init__(self, patience: int = 20, path: str = 'best_model.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None:
            self.best_score = score
            self._save(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path)


def _model_forward(model, data, model_type: str) -> torch.Tensor:
    """
    Dispatch forward call based on model type.

    Node-level classification: each graph is processed individually.
    DeepGNN / ShallowGNN expect (x, edge_index).
    HybridGNNTransformer expects (x, edge_index, pos).
    """
    if model_type == 'hybrid':
        return model(data.x, data.edge_index, data.pos)
    else:
        return model(data.x, data.edge_index)


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """
    Train for one epoch.

    Handles gradient clipping if specified in config.
    """
    model.train()
    total_loss = 0.0
    total_nodes = 0
    model_type = config['model']['type']

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = _model_forward(model, data, model_type)
        loss = criterion(out, data.y.long())
        loss.backward()

        if 'grad_clip_norm' in config.get('training', {}):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['training']['grad_clip_norm']
            )

        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
        total_nodes += data.y.size(0)

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, loader, device, config):
    """
    Evaluate model on a data loader.

    Returns:
        avg_loss, preds (np.ndarray), labels (np.ndarray), probs (np.ndarray)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_nodes = 0
    all_preds, all_labels, all_probs = [], [], []
    model_type = config['model']['type']

    for data in loader:
        data = data.to(device)
        out = _model_forward(model, data, model_type)
        loss = criterion(out, data.y.long())
        total_loss += loss.item() * data.y.size(0)
        total_nodes += data.y.size(0)

        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())
        all_probs.append(probs[:, 1].cpu())

    return (
        total_loss / max(total_nodes, 1),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )


def log_dirichlet_energy(model, data, edge_index, model_type: str):
    """Log Dirichlet energy per GNN layer for over-smoothing analysis."""
    layer_outputs = getattr(model, 'layer_outputs', None) or getattr(
        model, 'gnn_layer_outputs', None
    )
    if layer_outputs:
        energies = []
        for i, h in enumerate(layer_outputs):
            e = dirichlet_energy(h.detach(), edge_index)
            energies.append(e)
        return energies
    return []


def main():
    parser = argparse.ArgumentParser(description='Train pipeline crack detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--weights', type=str, default=None, help='Path to checkpoint weights to resume/continue training from')
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset ---
    dataset = CrackGraphDataset(config['data']['graph_dir'])
    train_idx, val_idx, test_idx = get_splits(
        dataset,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        seed,
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    batch_size = config['training'].get('batch_size', 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = build_model(config).to(device)
    print(f"Model type: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.weights and os.path.exists(args.weights):
        print(f"Loading checkpoint weights from {args.weights} to continue training...")
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # --- Loss Criterion ---
    loss_type = config['training'].get('loss', 'weighted_ce')
    gamma = float(config['training'].get('focal_gamma', 2.0))

    if loss_type == 'focal':
        alpha = get_inverted_class_weights(train_dataset).to(device)
        print(f"Using Focal Loss (gamma={gamma}, alpha=[{alpha[0]:.4f}, {alpha[1]:.4f}])")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'inverted_ce':
        weights = get_inverted_class_weights(train_dataset).to(device)
        weights = weights * 2.0
        print(f"Using Inverted Frequency Weighted CE (weights=[{weights[0]:.4f}, {weights[1]:.4f}])")
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        weights = get_class_weights(train_dataset).to(device)
        print(f"Using Standard Class-Weighted CE (weights=[{weights[0]:.4f}, {weights[1]:.4f}])")
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    warmup_epochs = config['training'].get('warmup_epochs', 0)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    scheduler_warmup = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if warmup_epochs > 0
        else None
    )
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # --- Training loop ---
    os.makedirs(config['output_dir'], exist_ok=True)
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        path=os.path.join(config['output_dir'], 'best_model.pt'),
    )

    train_losses, val_losses, val_f1_scores = [], [], []
    gate_history, temp_history = [], []

    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, device, config)

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_iou = compute_iou(val_preds, val_labels)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        # LR scheduling
        if scheduler_warmup and epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_plateau.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']
        early_stopping(val_f1, model)

        extra_info = []
        if hasattr(model, 'get_gate_weight') and model.get_gate_weight() is not None:
            gw = model.get_gate_weight()
            gate_history.append(gw)
            extra_info.append(f"Gate(Trans): {gw:.4f}")
        if hasattr(model, 'get_temperatures') and model.get_temperatures():
            temps = model.get_temperatures()
            temp_history.append(temps)
            extra_info.append(f"Temps: {[round(t, 3) for t in temps]}")

        extra_str = f" | {' | '.join(extra_info)}" if extra_info else ""
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6f}{extra_str}"
        )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # --- Final test evaluation ---
    model.load_state_dict(
        torch.load(os.path.join(config['output_dir'], 'best_model.pt'), map_location=device)
    )
    test_loss, test_preds, test_labels, test_probs = evaluate(model, test_loader, device, config)

    test_f1 = compute_f1(test_preds, test_labels)
    test_iou = compute_iou(test_preds, test_labels)
    test_auc = compute_roc_auc(test_probs, test_labels)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(classification_report(test_labels, test_preds, zero_division=0))
    print(f"F1: {test_f1:.4f} | IoU: {test_iou:.4f} | ROC-AUC: {test_auc:.4f}")
    if hasattr(model, 'get_gate_weight') and model.get_gate_weight() is not None:
        print(f"Final Best Checkpoint Gate(Transformer): {model.get_gate_weight():.4f} (GNN weight: {1.0 - model.get_gate_weight():.4f})")
    if hasattr(model, 'get_temperatures') and model.get_temperatures():
        print(f"Final Best Checkpoint Temperatures: {model.get_temperatures()}")

    # Save results
    results = {
        'test_f1': float(test_f1),
        'test_iou': float(test_iou),
        'test_roc_auc': float(test_auc),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_f1_scores': [float(x) for x in val_f1_scores],
        'gate_history': [float(x) for x in gate_history],
        'temp_history': [[float(t) for t in ts] for ts in temp_history],
    }
    if hasattr(model, 'get_gate_weight') and model.get_gate_weight() is not None:
        results['final_gate_transformer'] = float(model.get_gate_weight())
        results['final_gate_gnn'] = float(1.0 - model.get_gate_weight())
    if hasattr(model, 'get_temperatures') and model.get_temperatures():
        results['final_temperatures'] = [float(t) for t in model.get_temperatures()]

    import json

    with open(os.path.join(config['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {config['output_dir']}")


if __name__ == '__main__':
    main()
