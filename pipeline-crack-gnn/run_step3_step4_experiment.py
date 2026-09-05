import os, sys, json, time, glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc, dirichlet_energy
from src.train import load_config, get_inverted_class_weights, FocalLoss, EarlyStopping, train_one_epoch, evaluate

def train_and_eval(config_path, desc):
    print('='*70)
    print(f'Starting training: {desc}')
    print(f'Config: {config_path}')
    print('='*70)
    
    config = load_config(config_path)
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    dataset = CrackGraphDataset(config['data']['graph_dir'])
    train_idx, val_idx, test_idx = get_splits(dataset, config['data']['train_ratio'], config['data']['val_ratio'], seed)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training'].get('batch_size', 1), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model type: {config[" model\][\type\]}, params: {total_params:,}')
 
 alpha = get_inverted_class_weights(train_dataset).to(device)
 criterion = FocalLoss(alpha=alpha, gamma=config['training'].get('focal_gamma', 2.0))
 optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
 
 warmup_epochs = config['training'].get('warmup_epochs', 0)
 def lr_lambda(e):
 return float(e + 1) / float(warmup_epochs) if warmup_epochs > 0 and e < warmup_epochs else 1.0
 scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if warmup_epochs > 0 else None
 scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
 
 os.makedirs(config['output_dir'], exist_ok=True)
 best_ckpt_path = os.path.join(config['output_dir'], 'best_model.pt')
 early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], path=best_ckpt_path)
 
 gate_history, temp_history = [], []
 train_losses, val_losses, val_f1s = [], [], []
 
 start_time = time.time()
 for epoch in range(config['training']['epochs']):
 tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
 v_loss, v_preds, v_labels, v_probs = evaluate(model, val_loader, device, config)
 v_f1 = f1_score(v_labels, v_preds, zero_division=0)
 v_iou = compute_iou(v_preds, v_labels)
 
 train_losses.append(tr_loss)
 val_losses.append(v_loss)
 val_f1s.append(v_f1)
 
 if scheduler_warmup and epoch < warmup_epochs:
 scheduler_warmup.step()
 else:
 scheduler_plateau.step(v_f1)
 
 early_stopping(v_f1, model)
 
 gw = model.get_gate_weight() if hasattr(model, 'get_gate_weight') else None
 if gw is not None: gate_history.append(gw)
 temps = model.get_temperatures() if hasattr(model, 'get_temperatures') else []
 if temps: temp_history.append(temps)
 
 extra = []
 if gw is not None: extra.append(f'Gate(Trans): {gw:.4f}')
 if temps: extra.append(f'Temps: {[round(t, 3) for t in temps]}')
 estr = ' | ' + ' | '.join(extra) if extra else ''
 
 if (epoch + 1) % 5 == 0 or epoch < 5 or early_stopping.early_stop:
 print(f'Epoch {epoch+1:03d} | TrLoss: {tr_loss:.4f} | ValLoss: {v_loss:.4f} | ValF1: {v_f1:.4f} | ValIoU: {v_iou:.4f}{estr}')
 
 if early_stopping.early_stop:
 print(f'Early stopping triggered at epoch {epoch+1}.')
 break
 
 train_sec = time.time() - start_time
 print(f'Training finished in {train_sec:.2f}s.')
 
 # Load best checkpoint
 model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
 
 # Val evaluation
 _, _, val_labels_all, val_probs_all = evaluate(model, val_loader, device, config)
 
 # Test evaluation
 _, _, test_labels_all, test_probs_all = evaluate(model, test_loader, device, config)
 
 # Holdout benchmark evaluation
 holdout_ds = CrackGraphDataset('data/graphs/test')
 holdout_loader = DataLoader(holdout_ds, batch_size=1, shuffle=False)
 _, _, holdout_labels_all, holdout_probs_all = evaluate(model, holdout_loader, device, config)
 
 # Sweep threshold on val
 best_tau, best_val_f1 = 0.50, 0.0
 for tau in np.arange(0.10, 0.92, 0.02):
 preds = (val_probs_all >= tau).astype(int)
 f1 = f1_score(val_labels_all, preds, zero_division=0)
 if f1 > best_val_f1:
 best_val_f1 = f1
 best_tau = tau
 
 # Metrics calculator
 def calc_metrics(labels, probs, tau):
 preds = (probs >= tau).astype(int)
 f1 = f1_score(labels, preds, zero_division=0)
 p = precision_score(labels, preds, zero_division=0)
 r = recall_score(labels, preds, zero_division=0)
 iou = compute_iou(preds, labels)
 auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
 return {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'iou': float(iou), 'roc_auc': float(auc)}
 
 test_m_05 = calc_metrics(test_labels_all, test_probs_all, 0.50)
 test_m_opt = calc_metrics(test_labels_all, test_probs_all, best_tau)
 holdout_m_05 = calc_metrics(holdout_labels_all, holdout_probs_all, 0.50)
 holdout_m_opt = calc_metrics(holdout_labels_all, holdout_probs_all, best_tau)
 
 final_gw = model.get_gate_weight() if hasattr(model, 'get_gate_weight') else None
 final_temps = model.get_temperatures() if hasattr(model, 'get_temperatures') else []
 
 print(f'-- Results for {desc} --')
 print(f'Optimal Tau: {best_tau:.2f} (Val F1: {best_val_f1:.4f})')
 print(f'Holdout @ 0.50: F1={holdout_m_05[\f1\]:.4f}, IoU={holdout_m_05[\iou\]:.4f}, Recall={holdout_m_05[\recall\]:.4f}')
 print(f'Holdout @ {best_tau:.2f}: F1={holdout_m_opt[\f1\]:.4f}, IoU={holdout_m_opt[\iou\]:.4f}, Recall={holdout_m_opt[\recall\]:.4f}')
 if final_gw is not None:
 print(f'Final Gate Transformer Weight: {final_gw:.4f} (GNN: {1.0 - final_gw:.4f})')
 if final_temps:
 print(f'Final Temperatures: {[round(t, 3) for t in final_temps]}')
 
 res = {
 'model_desc': desc,
 'config_path': config_path,
 'train_time_sec': train_sec,
 'optimal_tau': float(best_tau),
 'best_val_f1': float(best_val_f1),
 'test_05': test_m_05,
 'test_opt': test_m_opt,
 'holdout_05': holdout_m_05,
 'holdout_opt': holdout_m_opt,
 'final_gate_transformer': float(final_gw) if final_gw is not None else None,
 'final_gate_gnn': float(1.0 - final_gw) if final_gw is not None else None,
 'final_temperatures': [float(t) for t in final_temps] if final_temps else None,
 'gate_history': [float(g) for g in gate_history],
 'temp_history': [[float(t) for t in ts] for ts in temp_history],
 'train_losses': [float(l) for l in train_losses],
 'val_losses': [float(l) for l in val_losses],
 'val_f1s': [float(f) for f in val_f1s]
 }
 with open(os.path.join(config['output_dir'], 'experiment_summary.json'), 'w') as f:
 json.dump(res, f, indent=2)
 return res, model

if __name__ == '__main__':
 res_step3, m3_model = train_and_eval('configs/step3_m4_gate.yaml', 'Step 3: M4 + Learned Residual Gate')
 res_step4, m4_model = train_and_eval('configs/step4_m4_relpos.yaml', 'Step 4: M4 + RelPos Bias + Residual Gate')
 print('\nFinished Step 3 and Step 4 training runs!')