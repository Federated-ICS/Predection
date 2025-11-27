#!/usr/bin/env python3
"""
Enhanced Training with Network-Aware Graph Structure
This version incorporates network features into the graph edges
"""

print("=" * 70)
print("NETWORK-AWARE HYBRID GNN - ENHANCED GRAPH STRUCTURE")
print("=" * 70)

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv

# For visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    PLOT_AVAILABLE = True
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
except ImportError:
    print("⚠️  matplotlib/seaborn not available. Install with: pip install matplotlib seaborn scikit-learn")
    PLOT_AVAILABLE = False

# Configuration
HIDDEN_DIM = 256
NUM_HEADS = 8
DROPOUT = 0.4
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
NUM_EPOCHS = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"Hidden Dim: {HIDDEN_DIM}, Heads: {NUM_HEADS}, Batch: {BATCH_SIZE}")

# ============================================================================
# DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("DATA ANALYSIS")
print("=" * 70)

data_path = Path('data/sequences_with_network_features.json')
with open(data_path) as f:
    training_pairs = json.load(f)

print(f"\n✅ Loaded {len(training_pairs)} training pairs")

# Variant distribution
variant_counts = Counter(p['variant_type'] for p in training_pairs)
print(f"\nVariant Distribution:")
for variant, count in variant_counts.items():
    print(f"  {variant:12s}: {count:5d} ({100*count/len(training_pairs):.1f}%)")

# Top transitions
transitions = [(p['current_attack'], p['next_attack']) for p in training_pairs]
transition_counts = Counter(transitions)
print(f"\nTop 10 Transitions:")
for (curr, nxt), count in transition_counts.most_common(10):
    print(f"  {curr} → {nxt}: {count}")

# Network features analysis
all_features = np.array([p['network_features'] for p in training_pairs])
print(f"\nNetwork Features (3s window):")
print(f"  Packet count - Mean: {all_features[:, 0].mean():.2f}, Std: {all_features[:, 0].std():.2f}")
print(f"  Byte count   - Mean: {all_features[:, 1].mean():.2f}, Std: {all_features[:, 1].std():.2f}")

# ============================================================================
# BUILD NETWORK-AWARE GRAPH
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING NETWORK-AWARE GRAPH")
print("=" * 70)

all_techniques = sorted(set(p['current_attack'] for p in training_pairs) | 
                       set(p['next_attack'] for p in training_pairs))
technique_to_idx = {tech: idx for idx, tech in enumerate(all_techniques)}
idx_to_technique = {idx: tech for tech, idx in technique_to_idx.items()}

num_techniques = len(all_techniques)
num_network_features = len(training_pairs[0]['network_features'])

print(f"\n✅ {num_techniques} techniques, {num_network_features} network features")

# Build graph with network feature statistics per edge
print("\n📊 Aggregating network features per transition...")
edge_features_dict = defaultdict(lambda: defaultdict(list))

for pair in training_pairs:
    curr = pair['current_attack']
    nxt = pair['next_attack']
    net_feat = pair['network_features']
    
    edge_features_dict[curr][nxt].append(net_feat)

# Create multi-dimensional edge features
print("📊 Creating multi-dimensional edge features...")
edge_list, edge_features = [], []
total_transitions = len(training_pairs)

for curr_tech, next_techs in edge_features_dict.items():
    curr_idx = technique_to_idx[curr_tech]
    
    for next_tech, net_feat_list in next_techs.items():
        next_idx = technique_to_idx[next_tech]
        
        # Aggregate network features for this transition
        net_feat_array = np.array(net_feat_list)
        
        # Create 6D edge feature vector
        edge_feature = [
            len(net_feat_list) / total_transitions,  # Transition probability
            net_feat_array[:, 0].mean(),             # Avg packet count
            net_feat_array[:, 0].std() + 1e-8,       # Std packet count
            net_feat_array[:, 1].mean(),             # Avg byte count
            net_feat_array[:, 1].std() + 1e-8,       # Std byte count
            np.log1p(len(net_feat_list))             # Log transition count
        ]
        
        edge_list.append([curr_idx, next_idx])
        edge_features.append(edge_feature)

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_features, dtype=torch.float)

# Normalize edge features
edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
edge_attr_std = edge_attr.std(dim=0, keepdim=True) + 1e-8
edge_attr = (edge_attr - edge_attr_mean) / edge_attr_std

edge_feature_dim = edge_attr.shape[1]

print(f"✅ Graph: {num_techniques} nodes, {edge_index.shape[1]} edges")
print(f"✅ Edge features: {edge_feature_dim}D (probability + network stats)")
print(f"\nEdge feature breakdown:")
print(f"  [0] Transition probability")
print(f"  [1] Avg packet count")
print(f"  [2] Std packet count")
print(f"  [3] Avg byte count")
print(f"  [4] Std byte count")
print(f"  [5] Log transition count")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

current_indices = torch.tensor([technique_to_idx[p['current_attack']] for p in training_pairs], dtype=torch.long)
next_indices = torch.tensor([technique_to_idx[p['next_attack']] for p in training_pairs], dtype=torch.long)
network_features_tensor = torch.tensor([p['network_features'] for p in training_pairs], dtype=torch.float)

# Normalize features
mean = network_features_tensor.mean(dim=0, keepdim=True)
std = network_features_tensor.std(dim=0, keepdim=True) + 1e-8
network_features_tensor = (network_features_tensor - mean) / std

# Split data
n = len(training_pairs)
indices = torch.randperm(n)
train_size, val_size = int(0.7 * n), int(0.15 * n)
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

print(f"\n✅ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ============================================================================
# NETWORK-AWARE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("DEFINING NETWORK-AWARE MODEL")
print("=" * 70)

class NetworkAwareHybridGNN(nn.Module):
    """
    Enhanced Hybrid GNN with network-aware graph edges.
    
    Key improvement: GAT layers now use multi-dimensional edge features
    that include network statistics (packet/byte counts) for each transition.
    """
    def __init__(self, num_nodes, num_network_features, edge_feature_dim=6, 
                 hidden_dim=256, num_heads=8, dropout=0.4):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # 3-layer network encoder with batch norm (for current observation)
        self.network_encoder = nn.Sequential(
            nn.Linear(num_network_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 3 GAT layers with edge features (NETWORK-AWARE!)
        self.gat1 = GATConv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            edge_dim=edge_feature_dim,  # Now uses edge features!
            dropout=dropout
        )
        self.gat2 = GATConv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            edge_dim=edge_feature_dim,  # Network-aware attention
            dropout=dropout
        )
        self.gat3 = GATConv(
            hidden_dim, 
            hidden_dim, 
            heads=1, 
            edge_dim=edge_feature_dim,  # Final layer also uses edge features
            dropout=dropout
        )
        
        # Fusion with batch norm
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_nodes)
        
    def forward(self, current_node_ids, network_features, edge_index, edge_attr):
        # Graph convolution with network-aware edges
        x = self.node_embedding.weight
        
        # GAT layers now use edge_attr (network statistics)
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat3(x, edge_index, edge_attr)
        
        # Extract embeddings for current nodes
        node_emb = x[current_node_ids]
        
        # Encode current network observation
        net_emb = self.network_encoder(network_features)
        
        # Fuse graph and network information
        combined = torch.cat([node_emb, net_emb], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits

model = NetworkAwareHybridGNN(
    num_techniques, 
    num_network_features, 
    edge_feature_dim,
    HIDDEN_DIM, 
    NUM_HEADS, 
    DROPOUT
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model: {num_params:,} parameters")
print(f"✅ Edge feature dimension: {edge_feature_dim}D")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

edge_index = edge_index.to(device)
edge_attr = edge_attr.to(device)

best_val_acc = 0
Path('models').mkdir(exist_ok=True)

# Track training history
train_losses, train_accs = [], []
val_losses, val_accs = [], []

print("\n🚀 Training started...\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    perm = torch.randperm(len(train_idx))
    shuffled_train_idx = train_idx[perm]
    
    epoch_loss, epoch_correct, num_batches = 0, 0, 0
    
    for i in range(0, len(shuffled_train_idx), BATCH_SIZE):
        batch_idx = shuffled_train_idx[i:i + BATCH_SIZE]
        curr = current_indices[batch_idx].to(device)
        nxt = next_indices[batch_idx].to(device)
        net_feat = network_features_tensor[batch_idx].to(device)
        
        optimizer.zero_grad()
        logits = model(curr, net_feat, edge_index, edge_attr)
        loss = criterion(logits, nxt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        epoch_correct += (predicted == nxt).sum().item()
        num_batches += 1
    
    train_loss = epoch_loss / num_batches
    train_acc = epoch_correct / len(train_idx)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    with torch.no_grad():
        curr = current_indices[val_idx].to(device)
        nxt = next_indices[val_idx].to(device)
        net_feat = network_features_tensor[val_idx].to(device)
        
        logits = model(curr, net_feat, edge_index, edge_attr)
        val_loss = criterion(logits, nxt)
        
        _, predicted = torch.max(logits, 1)
        val_acc = (predicted == nxt).float().mean().item()
        
        _, top3 = torch.topk(logits, 3, dim=1)
        val_top3 = sum((nxt[i] in top3[i]) for i in range(len(nxt))) / len(nxt)
        
        _, top5 = torch.topk(logits, 5, dim=1)
        val_top5 = sum((nxt[i] in top5[i]) for i in range(len(nxt))) / len(nxt)
    
    val_losses.append(val_loss.item())
    val_accs.append(val_acc)
    
    scheduler.step(val_loss.item())
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss.item():.4f}/{val_acc:.4f} Top3:{val_top3:.4f} Top5:{val_top5:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_network_aware_model.pt')
        print(f"    💾 Saved best model (val_acc: {best_val_acc:.4f})")

print(f"\n✅ Training complete! Best val: {best_val_acc:.4f}")

# ============================================================================
# TEST EVALUATION WITH METRICS
# ============================================================================
print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

model.load_state_dict(torch.load('models/best_network_aware_model.pt'))
model.eval()

with torch.no_grad():
    curr = current_indices[test_idx].to(device)
    nxt = next_indices[test_idx].to(device)
    net_feat = network_features_tensor[test_idx].to(device)
    
    logits = model(curr, net_feat, edge_index, edge_attr)
    test_loss = criterion(logits, nxt)
    
    _, predicted = torch.max(logits, 1)
    test_acc = (predicted == nxt).float().mean().item()
    
    _, top3 = torch.topk(logits, 3, dim=1)
    test_top3 = sum((nxt[i] in top3[i]) for i in range(len(nxt))) / len(nxt)
    
    _, top5 = torch.topk(logits, 5, dim=1)
    test_top5 = sum((nxt[i] in top5[i]) for i in range(len(nxt))) / len(nxt)

print(f"\n📊 Test Results:")
print(f"   Loss:      {test_loss.item():.4f}")
print(f"   Top-1 Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Top-3 Acc: {test_top3:.4f} ({test_top3*100:.2f}%)")
print(f"   Top-5 Acc: {test_top5:.4f} ({test_top5*100:.2f}%)")

# Per-class analysis
print(f"\n📈 Per-Class Analysis (Top 10 classes):")
predicted_cpu = predicted.cpu().numpy()
nxt_cpu = nxt.cpu().numpy()

class_correct = defaultdict(int)
class_total = defaultdict(int)

for pred, true in zip(predicted_cpu, nxt_cpu):
    class_total[true] += 1
    if pred == true:
        class_correct[true] += 1

sorted_classes = sorted(class_total.items(), key=lambda x: x[1], reverse=True)[:10]
for class_idx, total in sorted_classes:
    correct = class_correct[class_idx]
    acc = correct / total if total > 0 else 0
    tech_name = idx_to_technique[class_idx]
    print(f"   {tech_name:10s}: {correct:3d}/{total:3d} = {acc:.3f}")

# Save mappings and normalization parameters
mappings = {
    'technique_to_idx': technique_to_idx,
    'idx_to_technique': idx_to_technique,
    'num_techniques': num_techniques,
    'num_network_features': num_network_features,
    'edge_feature_dim': edge_feature_dim,
    'network_mean': mean.numpy(),
    'network_std': std.numpy(),
    'edge_attr_mean': edge_attr_mean.numpy(),
    'edge_attr_std': edge_attr_std.numpy()
}

with open('models/network_aware_mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

# Save training history
history = {
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs
}

with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print(f"\n✅ Model saved to: models/best_network_aware_model.pt")
print(f"✅ Mappings saved to: models/network_aware_mappings.pkl")
print(f"✅ Training history saved to: models/training_history.pkl")
print("\n" + "=" * 70)

# ============================================================================
# VISUALIZATION & ANALYSIS
# ============================================================================
if PLOT_AVAILABLE:
    print("\n" + "=" * 70)
    print("GENERATING FIGURES & ANALYSIS")
    print("=" * 70)
    
    # Create figures directory
    Path('figures').mkdir(exist_ok=True)
    
    # ========== FIGURE 1: Training Curves ==========
    print("\n📊 Creating training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, train_accs, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=3)
    axes[0, 1].plot(epochs, val_accs, 'r-', linewidth=2, label='Val Acc', marker='s', markersize=3)
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top-K Accuracy
    top_k_accs = [test_acc, test_top3, test_top5]
    top_k_labels = ['Top-1', 'Top-3', 'Top-5']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = axes[1, 0].bar(top_k_labels, top_k_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 0].set_title('Top-K Accuracy on Test Set', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, top_k_accs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.3f}\n({acc*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 4: Per-Class Accuracy (Top 10)
    sorted_classes_list = sorted(class_total.items(), key=lambda x: x[1], reverse=True)[:10]
    class_names = [idx_to_technique[idx][:10] for idx, _ in sorted_classes_list]
    class_accs = [class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0 
                  for idx, _ in sorted_classes_list]
    
    bars = axes[1, 1].barh(class_names, class_accs, color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_title('Per-Class Accuracy (Top 10 Classes)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    for bar, acc in zip(bars, class_accs):
        width = bar.get_width()
        axes[1, 1].text(width, bar.get_y() + bar.get_height()/2.,
                       f' {acc:.3f}',
                       ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/network_aware_training.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/network_aware_training.png")
    plt.close()
    
    # ========== FIGURE 2: Edge Feature Analysis ==========
    print("\n📊 Creating edge feature analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    edge_attr_np = edge_attr.cpu().numpy()
    feature_names = ['Trans. Prob', 'Avg Packets', 'Std Packets', 'Avg Bytes', 'Std Bytes', 'Log Count']
    
    for idx, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        ax.hist(edge_attr_np[:, idx], bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_val = edge_attr_np[:, idx].mean()
        std_val = edge_attr_np[:, idx].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=9)
    
    plt.suptitle('Network-Aware Edge Features Distribution', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('figures/edge_features_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/edge_features_distribution.png")
    plt.close()
    
    # ========== FIGURE 3: Network Feature Variance Analysis ==========
    print("\n📊 Creating network variance analysis...")
    
    # Analyze which transitions have high network variance
    high_variance_transitions = []
    
    for (curr_tech, next_tech), net_feat_list in edge_features_dict.items():
        if len(net_feat_list) >= 5:  # Only consider transitions with enough samples
            net_feat_array = np.array(net_feat_list)
            
            avg_packets = net_feat_array[:, 0].mean()
            std_packets = net_feat_array[:, 0].std()
            avg_bytes = net_feat_array[:, 1].mean()
            std_bytes = net_feat_array[:, 1].std()
            
            cv_packets = std_packets / (avg_packets + 1e-8)  # Coefficient of variation
            cv_bytes = std_bytes / (avg_bytes + 1e-8)
            
            high_variance_transitions.append({
                'transition': f"{curr_tech[:8]}→{next_tech[:8]}",
                'cv_packets': cv_packets,
                'cv_bytes': cv_bytes,
                'avg_packets': avg_packets,
                'std_packets': std_packets,
                'avg_bytes': avg_bytes,
                'std_bytes': std_bytes,
                'count': len(net_feat_list)
            })
    
    high_variance_transitions.sort(key=lambda x: x['cv_packets'], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top 15 transitions by packet variance
    top_15 = high_variance_transitions[:15]
    trans_names = [t['transition'] for t in top_15]
    cv_packets = [t['cv_packets'] for t in top_15]
    
    bars = axes[0].barh(trans_names, cv_packets, color='#FF6B6B', alpha=0.8, edgecolor='black')
    axes[0].set_title('Top 15 Transitions by Packet Count Variance', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Coefficient of Variation (CV)')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    for bar, cv in zip(bars, cv_packets):
        width = bar.get_width()
        axes[0].text(width, bar.get_y() + bar.get_height()/2.,
                    f' {cv:.2f}',
                    ha='left', va='center', fontweight='bold', fontsize=8)
    
    # Plot 2: Scatter plot - Packet variance vs Byte variance
    all_cv_packets = [t['cv_packets'] for t in high_variance_transitions]
    all_cv_bytes = [t['cv_bytes'] for t in high_variance_transitions]
    all_counts = [t['count'] for t in high_variance_transitions]
    
    scatter = axes[1].scatter(all_cv_packets, all_cv_bytes, 
                             s=[c*2 for c in all_counts], 
                             c=all_counts, cmap='viridis', 
                             alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_title('Network Variance Correlation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Packet Count CV')
    axes[1].set_ylabel('Byte Count CV')
    axes[1].grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Transition Count', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('figures/network_variance_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/network_variance_analysis.png")
    plt.close()
    
    # ========== FIGURE 4: Confusion Matrix ==========
    print("\n📊 Creating confusion matrix...")
    
    # Get top 15 most frequent classes
    top_15_classes = [idx for idx, _ in sorted(class_total.items(), key=lambda x: x[1], reverse=True)[:15]]
    
    # Filter predictions
    mask = np.isin(nxt_cpu, top_15_classes)
    filtered_true = nxt_cpu[mask]
    filtered_pred = predicted_cpu[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_15_classes)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax)
    
    class_labels = [idx_to_technique[idx][:10] for idx in top_15_classes]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels,
           yticklabels=class_labels,
           title='Confusion Matrix (Top 15 Classes, Normalized)',
           ylabel='True Technique',
           xlabel='Predicted Technique')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_normalized[i, j] > 0.01:
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=7)
    
    plt.tight_layout()
    plt.savefig('figures/network_aware_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/network_aware_confusion_matrix.png")
    plt.close()
    
    # ========== FIGURE 5: Prediction Confidence Analysis ==========
    print("\n📊 Creating prediction confidence analysis...")
    
    # Get prediction probabilities
    model.eval()
    with torch.no_grad():
        curr = current_indices[test_idx].to(device)
        nxt = next_indices[test_idx].to(device)
        net_feat = network_features_tensor[test_idx].to(device)
        
        logits = model(curr, net_feat, edge_index, edge_attr)
        probs = F.softmax(logits, dim=1)
        
        max_probs, _ = torch.max(probs, dim=1)
        max_probs_cpu = max_probs.cpu().numpy()
        
        correct_mask = (predicted == nxt).cpu().numpy()
        correct_probs = max_probs_cpu[correct_mask]
        incorrect_probs = max_probs_cpu[~correct_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Confidence distribution
    axes[0].hist(correct_probs, bins=30, alpha=0.7, label='Correct', color='#4ECDC4', edgecolor='black')
    axes[0].hist(incorrect_probs, bins=30, alpha=0.7, label='Incorrect', color='#FF6B6B', edgecolor='black')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Confidence (Max Probability)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    axes[0].axvline(correct_probs.mean(), color='#4ECDC4', linestyle='--', linewidth=2, 
                   label=f'Correct Mean: {correct_probs.mean():.3f}')
    axes[0].axvline(incorrect_probs.mean(), color='#FF6B6B', linestyle='--', linewidth=2,
                   label=f'Incorrect Mean: {incorrect_probs.mean():.3f}')
    
    # Plot 2: Calibration curve
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (max_probs_cpu >= confidence_bins[i]) & (max_probs_cpu < confidence_bins[i+1])
        if mask.sum() > 0:
            acc = correct_mask[mask].mean()
            bin_accuracies.append(acc)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    axes[1].plot(bin_centers, bin_accuracies, marker='o', linewidth=2, markersize=10, 
                color='#45B7D1', label='Model Calibration')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    axes[1].set_title('Model Calibration Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    # Add sample counts
    for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
        if count > 0:
            axes[1].text(x, y + 0.03, f'n={count}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/network_aware_confidence.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/network_aware_confidence.png")
    plt.close()
    
    # ========== FIGURE 6: Data Distribution Analysis ==========
    print("\n📊 Creating data distribution analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Variant Distribution
    variants = list(variant_counts.keys())
    counts = list(variant_counts.values())
    colors_var = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#FFE66D'][:len(variants)]
    
    wedges, texts, autotexts = axes[0, 0].pie(counts, labels=variants, autopct='%1.1f%%',
                                               colors=colors_var, startangle=90)
    axes[0, 0].set_title('Variant Type Distribution', fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Plot 2: Transition Frequency Distribution
    transition_freqs = [count for _, count in transition_counts.most_common()]
    axes[0, 1].hist(transition_freqs, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Transition Frequency Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_ylabel('Number of Transitions')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Network Feature Distribution (Packet Count)
    axes[1, 0].hist(all_features[:, 0], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Packet Count Distribution (3s window)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Packet Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axvline(all_features[:, 0].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {all_features[:, 0].mean():.1f}')
    axes[1, 0].legend()
    
    # Plot 4: Network Feature Distribution (Byte Count)
    axes[1, 1].hist(all_features[:, 1], bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Byte Count Distribution (3s window)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Byte Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axvline(all_features[:, 1].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {all_features[:, 1].mean():.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/data_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/data_distribution.png")
    plt.close()
    
    # ========== FIGURE 7: Graph Structure Visualization ==========
    print("\n📊 Creating graph structure visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Degree distribution
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        out_degrees[src] += 1
        in_degrees[dst] += 1
    
    in_deg_values = list(in_degrees.values())
    out_deg_values = list(out_degrees.values())
    
    axes[0].hist(in_deg_values, bins=30, alpha=0.7, label='In-degree', color='#4ECDC4', edgecolor='black')
    axes[0].hist(out_deg_values, bins=30, alpha=0.7, label='Out-degree', color='#FF6B6B', edgecolor='black')
    axes[0].set_title('Node Degree Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Number of Nodes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Edge weight distribution (transition probability)
    edge_probs = edge_attr[:, 0].cpu().numpy()  # First feature is transition probability
    axes[1].hist(edge_probs, bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[1].set_title('Edge Weight Distribution (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Normalized Transition Probability')
    axes[1].set_ylabel('Number of Edges')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axvline(edge_probs.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {edge_probs.mean():.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/graph_structure.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/graph_structure.png")
    plt.close()
    
    # ========== Save High Variance Transitions Report ==========
    print("\n📊 Saving high variance transitions report...")
    
    with open('figures/high_variance_transitions.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIGH VARIANCE TRANSITIONS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("These transitions show high variability in network features,\n")
        f.write("indicating they benefit most from network-aware graph edges.\n\n")
        
        f.write(f"{'Rank':<6}{'Transition':<30}{'CV Packets':<12}{'CV Bytes':<12}{'Count':<8}\n")
        f.write("-" * 80 + "\n")
        
        for i, trans in enumerate(high_variance_transitions[:30], 1):
            f.write(f"{i:<6}{trans['transition']:<30}{trans['cv_packets']:<12.3f}"
                   f"{trans['cv_bytes']:<12.3f}{trans['count']:<8}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        all_cv_packets = [t['cv_packets'] for t in high_variance_transitions]
        all_cv_bytes = [t['cv_bytes'] for t in high_variance_transitions]
        
        f.write(f"Total transitions analyzed: {len(high_variance_transitions)}\n")
        f.write(f"Packet CV - Mean: {np.mean(all_cv_packets):.3f}, Median: {np.median(all_cv_packets):.3f}\n")
        f.write(f"Byte CV - Mean: {np.mean(all_cv_bytes):.3f}, Median: {np.median(all_cv_bytes):.3f}\n")
        f.write(f"\nTransitions with CV > 0.5 (high variance): {sum(1 for cv in all_cv_packets if cv > 0.5)}\n")
        f.write(f"Transitions with CV > 1.0 (very high variance): {sum(1 for cv in all_cv_packets if cv > 1.0)}\n")
    
    print("✅ Saved: figures/high_variance_transitions.txt")
    
    print("\n" + "=" * 70)
    print("FIGURES SUMMARY")
    print("=" * 70)
    print("\n✅ Generated 7 evaluation figures:")
    print("   1. figures/network_aware_training.png - Training curves & accuracies")
    print("   2. figures/edge_features_distribution.png - Edge feature analysis")
    print("   3. figures/network_variance_analysis.png - Network variance patterns")
    print("   4. figures/network_aware_confusion_matrix.png - Confusion matrix")
    print("   5. figures/network_aware_confidence.png - Confidence analysis")
    print("   6. figures/data_distribution.png - Data distribution plots")
    print("   7. figures/graph_structure.png - Graph topology analysis")
    print("\n✅ Generated 1 analysis report:")
    print("   - figures/high_variance_transitions.txt - High variance transitions")
    print("\n" + "=" * 70)

else:
    print("\n⚠️  Visualization skipped (matplotlib not available)")
    print("   Install with: pip install matplotlib seaborn scikit-learn")

print("\n🎉 Network-aware training complete!")
print("\n💡 Key Improvements:")
print("   - Graph edges now contain 6D features (not just 1D frequency)")
print("   - GAT layers use network statistics for attention computation")
print("   - Model can distinguish high-traffic vs low-traffic transitions")
print("=" * 70)
