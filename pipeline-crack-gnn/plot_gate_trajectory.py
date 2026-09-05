"""
Learned Residual Gate & Temperature Trajectory Analysis Script.

Analyzes the full epoch-by-epoch training trajectory of the learned residual gate:
  x_out = sigma(gate) * x_trans + (1 - sigma(gate)) * x_gnn
and temperature parameter in Relative Positional Bias:
  attn_bias = -dist / temperature
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

step3_json = os.path.join(BASE_DIR, 'results', 'step3_gate', 'M4_hybrid_gate', 'results.json')
step4_json = os.path.join(BASE_DIR, 'results', 'step4_relpos', 'M4_hybrid_relpos', 'results.json')

with open(step3_json, 'r') as f:
    d3 = json.load(f)

with open(step4_json, 'r') as f:
    d4 = json.load(f)

gate3 = d3.get('gate_history', [])
gate4 = d4.get('gate_history', [])
temps4 = [t[0] for t in d4.get('temp_history', [])]

epochs3 = list(range(1, len(gate3) + 1))
epochs4 = list(range(1, len(gate4) + 1))

print("=== Step 3: Learned Residual Gate Trajectory Analysis ===")
print(f"Total Epochs: {len(gate3)}")
print(f"Initial sigma(gate): {gate3[0]:.4f} (at epoch 1)")
print(f"Peak sigma(gate): {max(gate3):.4f} (at epoch {gate3.index(max(gate3)) + 1})")
print(f"Min sigma(gate): {min(gate3):.4f} (at epoch {gate3.index(min(gate3)) + 1})")
print(f"Final sigma(gate): {gate3[-1]:.4f}")
print(f"Gate Range (Max - Min): {max(gate3) - min(gate3):.4f}")
print(f"Standard Deviation: {np.std(gate3):.5f}")

print("\n=== Step 4: Learned Gate & Temperature Analysis ===")
print(f"Step 4 Gate Range: {min(gate4):.4f} to {max(gate4):.4f}")
print(f"Step 4 Initial Temp: {temps4[0]:.4f} -> Final Temp: {temps4[-1]:.4f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Gate Trajectory
ax1.plot(epochs3, gate3, marker='o', linewidth=2.5, markersize=5, color='#9b59b6', label='Step 3: M4 + Gate Alone')
ax1.plot(epochs4, gate4, marker='s', linewidth=2.5, markersize=5, color='#e67e22', label='Step 4: M4 + RelPos + Gate')
ax1.axhline(0.50, color='gray', linestyle='--', linewidth=1.5, label='Initial Balance (0.50)')

ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Learned Gate Weight $\\sigma(\\mathrm{gate})$', fontsize=12, fontweight='bold')
ax1.set_title('Learned Residual Gate Trajectory (Step 3 vs. Step 4)', fontsize=13, fontweight='bold')
ax1.set_ylim(0.46, 0.54)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left', fontsize=10)

# Text annotation explaining lack of drift
ax1.text(0.05, 0.85, f"Step 3 Delta: {max(gate3)-min(gate3):.4f}\nStd Dev: {np.std(gate3):.5f}\n(Minimal drift from 0.50)", 
         transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Subplot 2: Temperature Decay
ax2.plot(epochs4, temps4, marker='^', linewidth=2.5, markersize=5, color='#2980b9', label='RelPos Attention Temperature')
ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Learned Temperature Parameter', fontsize=12, fontweight='bold')
ax2.set_title('Step 4: Spatial Attention Temperature Decay ($1.0 \\to 0.001$)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=10)

ax2.text(0.05, 0.25, f"Initial Temp: {temps4[0]:.3f}\nFinal Temp: {temps4[-1]:.4f}\n(Autonomous spatial sharpening)", 
         transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
out_plot = os.path.join(PLOTS_DIR, 'gate_trajectory_analysis.png')
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"\nSaved trajectory plot to {out_plot}")

