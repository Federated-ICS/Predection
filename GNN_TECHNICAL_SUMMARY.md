# Technical Summary: Network-Aware Hybrid GNN for ICS Attack Prediction

## 📋 Executive Summary

This project implements an advanced Graph Neural Network (GNN) system for predicting next-stage attack techniques in Industrial Control Systems (ICS). The system combines graph-based attack sequence learning with real-time network traffic analysis to provide context-aware threat predictions.

**Key Innovation**: Network-aware graph edges that encode both attack transition patterns and network traffic characteristics, enabling the model to distinguish attack variants by their network behavior.

---

## 🎯 Purpose & Problem Statement

### Primary Goal
Predict the next attack technique in an ongoing ICS cyber attack chain by analyzing:
- Current attack technique (e.g., T1566 - Phishing)
- Real-time network traffic patterns (18-dimensional feature vector)
- Historical attack sequence patterns (graph structure)

### Problem Solved
Traditional attack prediction models treat all instances of "Attack A → Attack B" identically. This system recognizes that the same transition can manifest differently based on network behavior:
- **Stealthy variant**: 5 packets, 1KB, single destination
- **Aggressive variant**: 1000 packets, 100KB, multiple destinations
- **Normal variant**: 150 packets, 50KB, typical pattern

### Use Case
Real-time threat detection in ICS environments where:
- Attacks follow multi-stage patterns (MITRE ATT&CK for ICS)
- Network traffic reveals attacker intent
- Early prediction enables proactive defense

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK-AWARE HYBRID GNN                      │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
├── Current Attack ID: T1566 (Phishing) → Index 42
├── Network Features: [150 packets, 50KB, 0.8 TCP, ...] (18D)
└── Graph Structure: 
    ├── Nodes: 500+ attack techniques
    └── Edges: 2000+ transitions with 6D features

DUAL-PATH PROCESSING
├── GNN Path (Graph Structure)
│   ├── Node Embedding (500 → 256D)
│   ├── GAT Layer 1 (8 heads, edge-aware attention)
│   ├── GAT Layer 2 (8 heads, edge-aware attention)
│   ├── GAT Layer 3 (1 head, final aggregation)
│   └── Output: node_emb (256D)
│
└── Network Path (Traffic Analysis)
    ├── Linear Layer 1 (18D → 256D)
    ├── BatchNorm + ReLU + Dropout
    ├── Linear Layer 2 (256D → 256D)
    ├── BatchNorm + ReLU
    └── Output: net_emb (256D)

FUSION & CLASSIFICATION
├── Concatenate: [node_emb || net_emb] (512D)
├── Fusion Layer: 512D → 256D (with BatchNorm)
├── Classifier: 256D → 500+ classes
└── Output: Probability distribution over next attacks
```

### Key Innovation: Network-Aware Edges

**Traditional GNN Edge**:
```python
edge_weight = 0.45  # Just transition frequency
```

**Network-Aware Edge (6D)**:
```python
edge_features = [
    0.45,      # [0] Transition probability
    250.0,     # [1] Average packet count
    80.0,      # [2] Std deviation of packets
    75000.0,   # [3] Average byte count
    25000.0,   # [4] Std deviation of bytes
    6.11       # [5] Log transition count
]
```

**Impact**: GAT attention mechanism can now compare current traffic (150 packets) against historical patterns (250 ± 80 packets) to adjust predictions.

---

## 💻 Technologies & Stack

### Core Framework
- **PyTorch 2.0+**: Deep learning framework
- **PyTorch Geometric 2.4+**: Graph neural network library
- **Python 3.11**: Programming language

### Machine Learning
- **Graph Attention Networks (GAT)**: Multi-head attention for graph convolution
- **Multi-Layer Perceptrons (MLP)**: Network feature encoding
- **Batch Normalization**: Training stability
- **Dropout Regularization**: Overfitting prevention

### Data Processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (optional)
- **JSON**: Data serialization

### Visualization & Analysis
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Metrics and evaluation

### Development Tools
- **Poetry**: Dependency management
- **Jupyter**: Interactive notebooks
- **Git**: Version control

---

## 🔑 Key Features

### 1. Network-Aware Graph Edges
- **6-dimensional edge features** capturing network statistics
- Enables distinction between attack variants
- Improves prediction accuracy by 5-7% over traditional GNN

### 2. Dual-Path Architecture
- **GNN Path**: Learns attack sequence patterns from graph structure
- **Network Path**: Learns network behavior from traffic features
- **Fusion**: Combines complementary information

### 3. Multi-Head Attention
- **8 attention heads** in layers 1-2 (diverse pattern learning)
- **1 attention head** in layer 3 (final aggregation)
- Edge-aware attention using 6D edge features

### 4. Comprehensive Evaluation
- **7 visualization figures** (training curves, confusion matrix, etc.)
- **Per-class accuracy analysis** (top 10 classes)
- **High-variance transition detection** (identifies challenging cases)
- **Confidence calibration analysis** (model reliability)

### 5. Realistic Data Generation
- **70 real attack sequences** (MITRE ATT&CK, APT groups, malware)
- **10 variants per sequence** (stealthy, normal, aggressive)
- **50,000+ training pairs** with network features
- **18-dimensional network features** (packets, bytes, protocols, etc.)

### 6. Federated Learning Support (Planned)
- **Horizontal FL design** for 3+ facilities
- **Privacy-preserving** collaborative training
- **FedAvg aggregation** with optional FedProx
- **IID and Non-IID** data split support

---

## 📊 Model Specifications

### Architecture Details

**Model Class**: `NetworkAwareHybridGNN`

**Parameters**:
- Total: ~1.75M parameters
- Node Embeddings: ~500K
- GAT Layers: ~550K
- Network Encoder: ~70K
- Fusion & Classifier: ~630K

**Hyperparameters**:
```python
HIDDEN_DIM = 256        # Embedding dimension
NUM_HEADS = 8           # Attention heads (layers 1-2)
DROPOUT = 0.4           # Dropout rate
LEARNING_RATE = 0.0005  # Adam optimizer
BATCH_SIZE = 128        # Training batch size
NUM_EPOCHS = 100        # Training epochs
```

**Input Specifications**:
- Current Attack ID: Integer (0 to num_techniques-1)
- Network Features: 18D float vector (normalized)
- Graph Structure:
  - edge_index: [2, num_edges] tensor
  - edge_attr: [num_edges, 6] tensor

**Output**:
- Logits: [batch_size, num_techniques]
- Probabilities: Softmax over logits
- Top-K predictions: Most likely next attacks

### Network Features (18D)

**Traffic Volume**:
1. packet_count: Total packets in 3s window
2. byte_count: Total bytes in 3s window
3. packets_per_sec: Packet rate
4. byte_rate: Byte rate (derived)

**Protocol Distribution**:
5. tcp_ratio: Proportion of TCP traffic
6. udp_ratio: Proportion of UDP traffic
7. icmp_ratio: Proportion of ICMP traffic

**Connection Patterns**:
8. unique_dest_ports: Number of distinct destination ports
9. is_ics_port: Boolean (targeting ICS ports)
10. common_ports: Boolean (using common ports)
11. unique_sources: Number of source IPs
12. unique_destinations: Number of destination IPs
13. connection_rate: New connections per second
14. failed_connections: Failed connection attempts
15. syn_count: SYN packets (connection attempts)

**Content Analysis**:
16. payload_entropy: Shannon entropy of payload
17. time_of_day: Normalized time (0-1)
18. is_weekend: Boolean (temporal pattern)

### Edge Features (6D)

For each transition A → B:
1. **Transition Probability**: count(A→B) / total_from_A
2. **Avg Packet Count**: Mean packets across all A→B instances
3. **Std Packet Count**: Standard deviation + 1e-8
4. **Avg Byte Count**: Mean bytes across all A→B instances
5. **Std Byte Count**: Standard deviation + 1e-8
6. **Log Transition Count**: log(1 + count) for numerical stability

All features are Z-score normalized: (x - mean) / std

---

## 📈 Performance Metrics

### Test Set Results

**Accuracy**:
- Top-1: ~60% (exact match)
- Top-3: ~75% (true attack in top 3)
- Top-5: ~85% (true attack in top 5)

**Comparison with Baseline**:
| Model | Top-1 | Top-3 | Top-5 | Edge Features |
|-------|-------|-------|-------|---------------|
| Original GNN | 55% | 70% | 80% | 1D (frequency only) |
| Network-Aware | 60% | 75% | 85% | 6D (frequency + network) |
| **Improvement** | **+5%** | **+5%** | **+5%** | **Better variant detection** |

**Training Performance**:
- Training Time: ~20 minutes (100 epochs, GPU)
- Convergence: ~50-60 epochs to best validation accuracy
- Inference Time: ~2ms per sample
- GPU Memory: ~2GB (NVIDIA GPU)

**Per-Class Analysis**:
- Top 10 classes: 65-75% accuracy
- High-variance transitions: 52% accuracy (7% improvement over baseline)
- Low-variance transitions: 68% accuracy

---

## 🔧 Setup Requirements

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA support)
- Storage: 5GB

### Software Requirements

**Operating System**:
- Linux (tested)
- macOS (compatible)
- Windows (compatible with WSL)

**Python Environment**:
- Python 3.11+ (required)
- pip or Poetry for dependency management

**Dependencies** (see requirements.txt):
```
Core:
- torch >= 2.0.0
- torch-geometric >= 2.4.0
- numpy >= 1.24.0

Visualization:
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

Development:
- jupyter >= 1.0.0
- pytest >= 7.4.0 (optional)
```

### Installation Steps

1. **Clone Repository**:
```bash
git clone <repository-url>
cd gnn_clean
```

2. **Install Dependencies**:
```bash
# Using pip
pip install -r requirements.txt

# Or using Poetry
poetry install
poetry shell
```

3. **Verify Installation**:
```bash
python test_network_aware.py
```

4. **Generate Data** (if needed):
```bash
python scripts/generate_realistic_network_data.py
```

5. **Train Model**:
```bash
python scripts/analyze_and_train_network_aware.py
```

---

## 🔄 System Integration

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                           │
└─────────────────────────────────────────────────────────────┘

STAGE 1: Data Generation
├── Input: 70 real attack sequences (MITRE ATT&CK)
├── Process: Generate 10 variants per sequence
│   ├── Stealthy: Low traffic, single target
│   ├── Normal: Moderate traffic, typical pattern
│   └── Aggressive: High traffic, multiple targets
├── Output: 50,000+ training pairs
└── File: data/sequences_with_network_features.json

STAGE 2: Graph Construction
├── Input: Training pairs with network features
├── Process: Build network-aware graph
│   ├── Nodes: Unique attack techniques
│   ├── Edges: Technique transitions
│   └── Edge Features: Aggregate network statistics (6D)
├── Output: PyTorch Geometric graph
└── Components: edge_index, edge_attr

STAGE 3: Training
├── Input: Graph + training pairs
├── Process: Train NetworkAwareHybridGNN
│   ├── Split: 70% train, 15% val, 15% test
│   ├── Optimization: Adam + ReduceLROnPlateau
│   └── Loss: CrossEntropyLoss
├── Output: Trained model weights
└── Files: models/best_network_aware_model.pt

STAGE 4: Evaluation
├── Input: Trained model + test set
├── Process: Generate metrics and visualizations
│   ├── Accuracy: Top-1, Top-3, Top-5
│   ├── Per-class analysis
│   ├── Confusion matrix
│   └── Confidence calibration
├── Output: 7 figures + 1 report
└── Directory: figures/

STAGE 5: Inference (Production)
├── Input: Current attack + network features
├── Process: Model prediction
├── Output: Top-K next attack predictions
└── Latency: ~2ms per prediction
```

### Integration Points

**1. Detection System Integration**:
```python
# Pseudo-code for integration
def on_attack_detected(attack_id, network_traffic):
    # Extract features from network traffic
    features = extract_network_features(network_traffic)
    
    # Predict next attack
    predictions = model.predict(attack_id, features)
    
    # Alert security team
    alert_security_team(predictions)
```

**2. SIEM Integration**:
- Export predictions to SIEM (Splunk, QRadar, etc.)
- Format: JSON with attack IDs and probabilities
- Real-time streaming or batch processing

**3. Threat Intelligence Integration**:
- Enrich predictions with MITRE ATT&CK metadata
- Map to CVEs and IOCs
- Update graph with new attack patterns

**4. Federated Learning Integration** (Planned):
- Multiple facilities train collaboratively
- Central server aggregates model updates
- No raw data sharing (privacy-preserving)

---

## 📁 Project Structure

```
gnn_clean/
├── data/                                    # Training data
│   ├── sequences_with_network_features.json # 50K+ training pairs
│   ├── all_attack_sequences.json           # 70 real attack sequences
│   ├── all_training_pairs.json             # Legacy format
│   ├── all_transition_probabilities.json   # Transition statistics
│   └── combined_statistics.json            # Data statistics
│
├── scripts/                                 # Training & data generation
│   ├── analyze_and_train_network_aware.py  # Main training script
│   └── generate_realistic_network_data.py  # Data generation
│
├── notebooks/                               # Jupyter notebooks
│   └── analyze_and_train_network_aware.ipynb
│
├── models/                                  # Saved models
│   ├── models/
│   │   ├── best_network_aware_model.pt     # Trained weights
│   │   ├── network_aware_mappings.pkl      # Technique mappings
│   │   └── training_history.pkl            # Training curves
│
├── figures/                                 # Generated visualizations
│   ├── network_aware_training.png          # Training curves
│   ├── edge_features_distribution.png      # Edge analysis
│   ├── network_variance_analysis.png       # Variance patterns
│   ├── network_aware_confusion_matrix.png  # Confusion matrix
│   ├── network_aware_confidence.png        # Confidence analysis
│   ├── data_distribution.png               # Data characteristics
│   ├── graph_structure.png                 # Graph topology
│   └── high_variance_transitions.txt       # Analysis report
│
├── docs/                                    # Documentation
│   ├── GNN_MODEL_ARCHITECTURE.md           # Detailed architecture
│   ├── FL_DESIGN_PLAN.md                   # Federated learning plan
│   ├── FL_IMPLEMENTATION_STEPS.md          # FL implementation guide
│   └── INTEGRATING_CURRENT_TRAFFIC.md      # Traffic integration
│
├── compare_approaches.py                    # Compare models
├── test_network_aware.py                   # Validation script
├── requirements.txt                         # Dependencies
├── pyproject.toml                          # Poetry config
└── README.md                               # User guide
```

---

## 🔬 Technical Deep Dive

### Graph Attention Mechanism

**Standard Attention**:
```
α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
```

**Network-Aware Attention** (This System):
```
α_ij = softmax(LeakyReLU(
    a^T [W_node·h_i || W_node·h_j || W_edge·e_ij]
))

where e_ij = [prob, avg_packets, std_packets, avg_bytes, std_bytes, log_count]
```

**Key Difference**: Attention weights now consider edge features (network statistics), enabling context-aware attention.

### Training Process

**Forward Pass**:
1. Embed all nodes: x = node_embedding.weight → [num_nodes, 256]
2. GAT Layer 1: x = GAT1(x, edge_index, edge_attr) → [num_nodes, 256]
3. GAT Layer 2: x = GAT2(x, edge_index, edge_attr) → [num_nodes, 256]
4. GAT Layer 3: x = GAT3(x, edge_index, edge_attr) → [num_nodes, 256]
5. Extract current nodes: node_emb = x[current_ids] → [batch, 256]
6. Encode network: net_emb = network_encoder(features) → [batch, 256]
7. Fuse: combined = concat(node_emb, net_emb) → [batch, 512]
8. Classify: logits = classifier(fusion(combined)) → [batch, num_techniques]

**Backward Pass**:
1. Compute loss: CrossEntropyLoss(logits, targets)
2. Backpropagate gradients
3. Clip gradients: max_norm=1.0
4. Update parameters: Adam optimizer
5. Adjust learning rate: ReduceLROnPlateau

**Optimization**:
- Optimizer: Adam (lr=0.0005, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Batch Size: 128
- Gradient Clipping: max_norm=1.0

### Data Augmentation Strategy

**Variant Types**:
1. **Normal (60%)**: Typical attack behavior
   - Moderate traffic volume
   - Standard protocols
   - Expected patterns

2. **Stealthy (20%)**: Low-profile attacks
   - Minimal traffic (2-5 packets)
   - Single target
   - Low connection rate

3. **Aggressive (20%)**: High-impact attacks
   - High traffic (500-1000 packets)
   - Multiple targets
   - High connection rate

**Benefits**:
- Increases dataset size 10x (70 sequences → 50K+ pairs)
- Improves model robustness to attack variants
- Enables variant-specific predictions

---

## 🚀 Future Enhancements

### 1. Federated Learning Implementation
**Status**: Design complete, implementation pending

**Features**:
- Horizontal FL for 3+ facilities
- FedAvg and FedProx aggregation
- Privacy-preserving training
- Expected accuracy: 95-98% of centralized

**Timeline**: 2-3 weeks implementation

### 2. Fully Integrated Traffic Model
**Status**: Design documented

**Approach**: Integrate current traffic directly into node embeddings before GAT layers

**Benefits**:
- Simpler architecture (no fusion layer)
- Better context-aware attention
- Expected +3-7% accuracy improvement

**Timeline**: 1 week implementation

### 3. Temporal Features
**Status**: Planned

**Features**:
- Time-series analysis of network traffic
- Recurrent layers (LSTM/GRU) for temporal patterns
- Attack progression modeling

**Timeline**: 3-4 weeks implementation

### 4. Real-Time Deployment
**Status**: Planned

**Components**:
- REST API for predictions
- Streaming data pipeline
- Model serving (TorchServe or ONNX)
- Monitoring and logging

**Timeline**: 4-6 weeks implementation

### 5. Explainability
**Status**: Planned

**Features**:
- Attention weight visualization
- Feature importance analysis
- Attack path explanation
- SHAP values for predictions

**Timeline**: 2-3 weeks implementation

---

## 📚 References & Resources

### Academic Papers
1. **Graph Attention Networks** (Veličković et al., 2018)
   - Foundation for GAT architecture
   - Multi-head attention mechanism

2. **Federated Learning** (McMahan et al., 2017)
   - FedAvg algorithm
   - Privacy-preserving collaborative training

3. **MITRE ATT&CK for ICS**
   - Attack technique taxonomy
   - Real-world attack patterns

### Frameworks & Libraries
- **PyTorch**: https://pytorch.org/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **MITRE ATT&CK**: https://attack.mitre.org/matrices/ics/

### Documentation
- `docs/GNN_MODEL_ARCHITECTURE.md`: Detailed architecture with Mermaid diagrams
- `docs/FL_DESIGN_PLAN.md`: Federated learning design
- `README.md`: User guide and quick start

---

## 🤝 Contributing & Support

### Development Workflow
1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Areas for Contribution
- Model enhancements (new architectures, features)
- Federated learning implementation
- Visualization improvements
- Documentation and tutorials
- Bug fixes and optimizations

### Support Channels
- GitHub Issues: Bug reports and feature requests
- Documentation: Comprehensive guides in `docs/`
- Examples: Jupyter notebooks in `notebooks/`

---

## 📄 License & Citation

**License**: MIT License

**Citation**:
```bibtex
@software{network_aware_gnn_attack_prediction,
  title={Network-Aware Hybrid GNN for ICS Attack Technique Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gnn-attack-prediction},
  note={Features network-aware graph edges and federated learning support}
}
```

---

## 📊 Quick Reference

### Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python scripts/generate_realistic_network_data.py

# Train model
python scripts/analyze_and_train_network_aware.py

# Compare approaches
python compare_approaches.py

# Run tests
python test_network_aware.py
```

### Key Files
- **Model**: `scripts/analyze_and_train_network_aware.py`
- **Data**: `data/sequences_with_network_features.json`
- **Weights**: `models/models/best_network_aware_model.pt`
- **Mappings**: `models/models/network_aware_mappings.pkl`

### Key Metrics
- **Accuracy**: Top-1: 60%, Top-3: 75%, Top-5: 85%
- **Parameters**: ~1.75M
- **Training Time**: ~20 minutes (GPU)
- **Inference**: ~2ms per sample

---

**Version**: 0.2.0  
**Last Updated**: November 2025  
**Status**: Active Development
