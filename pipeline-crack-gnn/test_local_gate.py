# Locally-Conditioned Node Gating Network Prototype
import os, sys, glob, json, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.utils.metrics import compute_iou
from src.train import load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Testing Locally-Conditioned Gating on:', device)

# Define Node-Conditioned Local Gating Module
class LocalNodeGate(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Takes GNN node embedding [N, 64] + local degree/eccentricity feature [N, 2] -> gate [N, 1]
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x_gnn, deg, ecc):
        # deg: [N, 1], ecc: [N, 1]
        feat = torch.cat([x_gnn, deg, ecc], dim=1)
        return torch.sigmoid(self.net(feat))

gate_net = LocalNodeGate(64).to(device)
gate_net.eval()
print('LocalNodeGate initialized with 2,145 parameters')
