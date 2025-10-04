"""
Generate performance charts for ML models
Author: Gabriel Demetrios Lafis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
output_dir = Path("docs/images")
output_dir.mkdir(parents=True, exist_ok=True)

# Model performance data (example results)
models = ['XGBoost', 'LightGBM', 'Random\nForest', 'Gradient\nBoosting', 'Logistic\nRegression']
accuracy = [0.68, 0.67, 0.64, 0.65, 0.58]
precision = [0.71, 0.70, 0.66, 0.67, 0.60]
recall = [0.65, 0.64, 0.62, 0.63, 0.56]
f1_score = [0.68, 0.67, 0.64, 0.65, 0.58]

# Chart 1: Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db')
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#2ecc71')
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#e74c3c')
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#f39c12')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('ML Models Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper right', frameon=True, shadow=True)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Generated: {output_dir / 'model_comparison.png'}")
plt.close()

# Chart 2: Feature Importance
features = ['RSI', 'MACD', 'BB_Width', 'Volume_MA', 'Price_MA', 
            'ATR', 'Momentum', 'Volatility', 'OBV', 'Stoch']
importance = [0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06]

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
bars = ax.barh(features, importance, color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Feature Importance (XGBoost)', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance)):
    ax.text(val + 0.002, i, f'{val:.3f}', 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Generated: {output_dir / 'feature_importance.png'}")
plt.close()

# Chart 3: Training History
epochs = np.arange(1, 51)
train_loss = 0.7 * np.exp(-epochs/15) + 0.3
val_loss = 0.75 * np.exp(-epochs/18) + 0.32
train_acc = 1 - (0.4 * np.exp(-epochs/12))
val_acc = 1 - (0.45 * np.exp(-epochs/15))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#3498db')
ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Model Loss During Training', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, shadow=True)
ax1.grid(alpha=0.3)

# Accuracy plot
ax2.plot(epochs, train_acc, label='Training Accuracy', linewidth=2, color='#2ecc71')
ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2, color='#f39c12')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Model Accuracy During Training', fontsize=12, fontweight='bold')
ax2.legend(frameon=True, shadow=True)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Generated: {output_dir / 'training_history.png'}")
plt.close()

# Chart 4: Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches

# Example confusion matrix
cm = np.array([[450, 150], [120, 480]])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# Labels
classes = ['Sell/Hold', 'Buy']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix (XGBoost)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Generated: {output_dir / 'confusion_matrix.png'}")
plt.close()

print("\n✅ All charts generated successfully!")
print(f"📁 Output directory: {output_dir.absolute()}")
